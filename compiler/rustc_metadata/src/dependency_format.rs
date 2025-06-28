//! Resolution of mixing rlibs and dylibs
//!
//! When producing a final artifact, such as a dynamic library, the compiler has
//! a choice between linking an rlib or linking a dylib of all upstream
//! dependencies. The linking phase must guarantee, however, that a library only
//! show up once in the object file. For example, it is illegal for library A to
//! be statically linked to B and C in separate dylibs, and then link B and C
//! into a crate D (because library A appears twice).
//!
//! The job of this module is to calculate what format each upstream crate
//! should be used when linking each output type requested in this session. This
//! generally follows this set of rules:
//!
//! 1. Each library must appear exactly once in the output.
//! 2. Each rlib contains only one library (it's just an object file)
//! 3. Each dylib can contain more than one library (due to static linking),
//!    and can also bring in many dynamic dependencies.
//!
//! With these constraints in mind, it's generally a very difficult problem to
//! find a solution that's not "all rlibs" or "all dylibs". I have suspicions
//! that NP-ness may come into the picture here...
//!
//! The current selection algorithm below looks mostly similar to:
//!
//! 1. If static linking is required, then require all upstream dependencies
//!    to be available as rlibs. If not, generate an error.
//! 2. If static linking is requested (generating an executable), then
//!    attempt to use all upstream dependencies as rlibs. If any are not
//!    found, bail out and continue to step 3.
//! 3. Static linking has failed, at least one library must be dynamically
//!    linked. Apply a heuristic by greedily maximizing the number of
//!    dynamically linked libraries.
//! 4. Each upstream dependency available as a dynamic library is
//!    registered. The dependencies all propagate, adding to a map. It is
//!    possible for a dylib to add a static library as a dependency, but it
//!    is illegal for two dylibs to add the same static library as a
//!    dependency. The same dylib can be added twice. Additionally, it is
//!    illegal to add a static dependency when it was previously found as a
//!    dylib (and vice versa)
//! 5. After all dynamic dependencies have been traversed, re-traverse the
//!    remaining dependencies and add them statically (if they haven't been
//!    added already).
//!
//! While not perfect, this algorithm should help support use-cases such as leaf
//! dependencies being static while the larger tree of inner dependencies are
//! all dynamic. This isn't currently very well battle tested, so it will likely
//! fall short in some use cases.
//!
//! Currently, there is no way to specify the preference of linkage with a
//! particular library (other than a global dynamic/static switch).
//! Additionally, the algorithm is geared towards finding *any* solution rather
//! than finding a number of solutions (there are normally quite a few).

use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_hir::def_id::{CrateNum, LOCAL_CRATE};
use rustc_index::IndexVec;
use rustc_middle::bug;
use rustc_middle::middle::dependency_format::{Dependencies, DependencyList, Linkage};
use rustc_middle::ty::TyCtxt;
use rustc_session::config::CrateType;
use rustc_session::cstore::CrateDepKind;
use rustc_session::cstore::LinkagePreference::{self, RequireDynamic, RequireStatic};
use rustc_span::sym;
use tracing::info;

use crate::creader::CStore;
use crate::errors::{
    BadPanicStrategy, CrateDepMultiple, IncompatiblePanicInDropStrategy, LibRequired,
    NonStaticCrateDep, RequiredPanicStrategy, RlibRequired, RustcDriverHelp, RustcLibRequired,
    TwoPanicRuntimes,
};

pub(crate) fn calculate(tcx: TyCtxt<'_>) -> Dependencies {
    tcx.crate_types()
        .iter()
        .map(|&ty| {
            let linkage = calculate_type(tcx, ty);
            verify_ok(tcx, &linkage);
            (ty, linkage)
        })
        .collect()
}

fn calculate_type(tcx: TyCtxt<'_>, ty: CrateType) -> DependencyList {
    let sess = &tcx.sess;

    if !sess.opts.output_types.should_link() {
        return IndexVec::new();
    }

    let preferred_linkage =
        match ty {
            // Generating a dylib without `-C prefer-dynamic` means that we're going
            // to try to eagerly statically link all dependencies. This is normally
            // done for end-product dylibs, not intermediate products.
            //
            // Treat cdylibs and staticlibs similarly. If `-C prefer-dynamic` is set,
            // the caller may be code-size conscious, but without it, it makes sense
            // to statically link a cdylib or staticlib. For staticlibs we use
            // `-Z staticlib-prefer-dynamic` for now. This may be merged into
            // `-C prefer-dynamic` in the future.
            CrateType::Dylib | CrateType::Cdylib | CrateType::Sdylib => {
                if sess.opts.cg.prefer_dynamic { Linkage::Dynamic } else { Linkage::Static }
            }
            CrateType::Staticlib => {
                if sess.opts.unstable_opts.staticlib_prefer_dynamic {
                    Linkage::Dynamic
                } else {
                    Linkage::Static
                }
            }

            // If the global prefer_dynamic switch is turned off, or the final
            // executable will be statically linked, prefer static crate linkage.
            CrateType::Executable if !sess.opts.cg.prefer_dynamic || sess.crt_static(Some(ty)) => {
                Linkage::Static
            }
            CrateType::Executable => Linkage::Dynamic,

            // proc-macro crates are mostly cdylibs, but we also need metadata.
            CrateType::ProcMacro => Linkage::Static,

            // No linkage happens with rlibs, we just needed the metadata (which we
            // got long ago), so don't bother with anything.
            CrateType::Rlib => Linkage::NotLinked,
        };

    let mut unavailable_as_static = Vec::new();

    match preferred_linkage {
        // If the crate is not linked, there are no link-time dependencies.
        Linkage::NotLinked => return IndexVec::new(),
        Linkage::Static => {
            // Attempt static linkage first. For dylibs and executables, we may be
            // able to retry below with dynamic linkage.
            if let Some(v) = attempt_static(tcx, &mut unavailable_as_static) {
                return v;
            }

            // Static executables must have all static dependencies.
            // If any are not found, generate some nice pretty errors.
            if (ty == CrateType::Staticlib && !sess.opts.unstable_opts.staticlib_allow_rdylib_deps)
                || (ty == CrateType::Executable
                    && sess.crt_static(Some(ty))
                    && !sess.target.crt_static_allows_dylibs)
            {
                for &cnum in tcx.crates(()).iter() {
                    if tcx.dep_kind(cnum).macros_only() {
                        continue;
                    }
                    let src = tcx.used_crate_source(cnum);
                    if src.rlib.is_some() {
                        continue;
                    }
                    sess.dcx().emit_err(RlibRequired { crate_name: tcx.crate_name(cnum) });
                }
                return IndexVec::new();
            }
        }
        Linkage::Dynamic | Linkage::IncludedFromDylib => {}
    }

    let all_dylibs = || {
        tcx.crates(()).iter().filter(|&&cnum| {
            !tcx.dep_kind(cnum).macros_only()
                && (tcx.used_crate_source(cnum).dylib.is_some()
                    || tcx.used_crate_source(cnum).sdylib_interface.is_some())
        })
    };

    let mut upstream_in_dylibs = FxHashSet::default();

    if tcx.features().rustc_private() {
        // We need this to prevent users of `rustc_driver` from linking dynamically to `std`
        // which does not work as `std` is also statically linked into `rustc_driver`.

        // Find all libraries statically linked to upstream dylibs.
        for &cnum in all_dylibs() {
            let deps = tcx.dylib_dependency_formats(cnum);
            for &(depnum, style) in deps.iter() {
                if let RequireStatic = style {
                    upstream_in_dylibs.insert(depnum);
                }
            }
        }
    }

    let mut formats = FxHashMap::default();

    // Sweep all crates for found dylibs. Add all dylibs, as well as their
    // dependencies, ensuring there are no conflicts. The only valid case for a
    // dependency to be relied upon twice is for both cases to rely on a dylib.
    for &cnum in all_dylibs() {
        if upstream_in_dylibs.contains(&cnum) {
            info!("skipping dylib: {}", tcx.crate_name(cnum));
            // If this dylib is also available statically linked to another dylib
            // we try to use that instead.
            continue;
        }

        let name = tcx.crate_name(cnum);
        info!("adding dylib: {}", name);
        add_library(tcx, cnum, RequireDynamic, &mut formats, &mut unavailable_as_static);
        let deps = tcx.dylib_dependency_formats(cnum);
        for &(depnum, style) in deps.iter() {
            info!("adding {:?}: {}", style, tcx.crate_name(depnum));
            add_library(tcx, depnum, style, &mut formats, &mut unavailable_as_static);
        }
    }

    // Collect what we've got so far in the return vector.
    let last_crate = tcx.crates(()).len();
    let mut ret = IndexVec::new();

    // We need to fill in something for LOCAL_CRATE as IndexVec is a dense map.
    // Linkage::Static semantically the most correct thing to use as the local
    // crate is always statically linked into the linker output, even when
    // linking a dylib. Using Linkage::Static also allow avoiding special cases
    // for LOCAL_CRATE in some places.
    assert_eq!(ret.push(Linkage::Static), LOCAL_CRATE);

    for cnum in 1..last_crate + 1 {
        let cnum = CrateNum::new(cnum);
        assert_eq!(
            ret.push(match formats.get(&cnum) {
                Some(&RequireDynamic) => Linkage::Dynamic,
                Some(&RequireStatic) => Linkage::IncludedFromDylib,
                None => Linkage::NotLinked,
            }),
            cnum
        );
    }

    // Run through the dependency list again, and add any missing libraries as
    // static libraries.
    //
    // If the crate hasn't been included yet and it's not actually required
    // (e.g., it's a panic runtime) then we skip it here as well.
    for &cnum in tcx.crates(()).iter() {
        let src = tcx.used_crate_source(cnum);
        if src.dylib.is_none()
            && !formats.contains_key(&cnum)
            && tcx.dep_kind(cnum) == CrateDepKind::Explicit
        {
            assert!(src.rlib.is_some() || src.rmeta.is_some());
            info!("adding staticlib: {}", tcx.crate_name(cnum));
            add_library(tcx, cnum, RequireStatic, &mut formats, &mut unavailable_as_static);
            ret[cnum] = Linkage::Static;
        }
    }

    // We've gotten this far because we're emitting some form of a final
    // artifact which means that we may need to inject dependencies of some
    // form.
    //
    // Things like panic runtimes may not have been activated quite yet, so do so here.
    activate_injected_dep(CStore::from_tcx(tcx).injected_panic_runtime(), &mut ret, &|cnum| {
        tcx.is_panic_runtime(cnum)
    });

    // When dylib B links to dylib A, then when using B we must also link to A.
    // It could be the case, however, that the rlib for A is present (hence we
    // found metadata), but the dylib for A has since been removed.
    //
    // For situations like this, we perform one last pass over the dependencies,
    // making sure that everything is available in the requested format.
    for (cnum, kind) in ret.iter_enumerated() {
        if cnum == LOCAL_CRATE {
            continue;
        }
        let src = tcx.used_crate_source(cnum);
        match *kind {
            Linkage::NotLinked | Linkage::IncludedFromDylib => {}
            Linkage::Static if src.rlib.is_some() => continue,
            Linkage::Dynamic if src.dylib.is_some() || src.sdylib_interface.is_some() => continue,
            kind => {
                let kind = match kind {
                    Linkage::Static => "rlib",
                    _ => "dylib",
                };
                let crate_name = tcx.crate_name(cnum);
                if crate_name.as_str().starts_with("rustc_") {
                    sess.dcx().emit_err(RustcLibRequired { crate_name, kind });
                } else {
                    sess.dcx().emit_err(LibRequired { crate_name, kind });
                }
            }
        }
    }

    ret
}

fn add_library(
    tcx: TyCtxt<'_>,
    cnum: CrateNum,
    link: LinkagePreference,
    m: &mut FxHashMap<CrateNum, LinkagePreference>,
    unavailable_as_static: &mut Vec<CrateNum>,
) {
    match m.get(&cnum) {
        Some(&link2) => {
            // If the linkages differ, then we'd have two copies of the library
            // if we continued linking. If the linkages are both static, then we
            // would also have two copies of the library (static from two
            // different locations).
            //
            // This error is probably a little obscure, but I imagine that it
            // can be refined over time.
            if link2 != link || link == RequireStatic {
                let linking_to_rustc_driver = tcx.sess.psess.unstable_features.is_nightly_build()
                    && tcx.crates(()).iter().any(|&cnum| tcx.crate_name(cnum) == sym::rustc_driver);
                tcx.dcx().emit_err(CrateDepMultiple {
                    crate_name: tcx.crate_name(cnum),
                    non_static_deps: unavailable_as_static
                        .drain(..)
                        .map(|cnum| NonStaticCrateDep { crate_name_: tcx.crate_name(cnum) })
                        .collect(),
                    rustc_driver_help: linking_to_rustc_driver.then_some(RustcDriverHelp),
                });
            }
        }
        None => {
            m.insert(cnum, link);
        }
    }
}

fn attempt_static(tcx: TyCtxt<'_>, unavailable: &mut Vec<CrateNum>) -> Option<DependencyList> {
    let all_crates_available_as_rlib = tcx
        .crates(())
        .iter()
        .copied()
        .filter_map(|cnum| {
            if tcx.dep_kind(cnum).macros_only() {
                return None;
            }
            let is_rlib = tcx.used_crate_source(cnum).rlib.is_some();
            if !is_rlib {
                unavailable.push(cnum);
            }
            Some(is_rlib)
        })
        .all(|is_rlib| is_rlib);
    if !all_crates_available_as_rlib {
        return None;
    }

    // All crates are available in an rlib format, so we're just going to link
    // everything in explicitly so long as it's actually required.
    let mut ret = IndexVec::new();
    assert_eq!(ret.push(Linkage::Static), LOCAL_CRATE);
    for &cnum in tcx.crates(()) {
        assert_eq!(
            ret.push(match tcx.dep_kind(cnum) {
                CrateDepKind::Explicit => Linkage::Static,
                CrateDepKind::MacrosOnly | CrateDepKind::Implicit => Linkage::NotLinked,
            }),
            cnum
        );
    }

    // Our panic runtime may not have been linked above if it wasn't explicitly
    // linked, which is the case for any injected dependency. Handle that here
    // and activate it.
    activate_injected_dep(CStore::from_tcx(tcx).injected_panic_runtime(), &mut ret, &|cnum| {
        tcx.is_panic_runtime(cnum)
    });

    Some(ret)
}

/// Given a list of how to link upstream dependencies so far, ensure that an
/// injected dependency is activated. This will not do anything if one was
/// transitively included already (e.g., via a dylib or explicitly so).
///
/// If an injected dependency was not found then we're guaranteed the
/// metadata::creader module has injected that dependency (not listed as
/// a required dependency) in one of the session's field. If this field is not
/// set then this compilation doesn't actually need the dependency and we can
/// also skip this step entirely.
fn activate_injected_dep(
    injected: Option<CrateNum>,
    list: &mut DependencyList,
    replaces_injected: &dyn Fn(CrateNum) -> bool,
) {
    for (cnum, slot) in list.iter_enumerated() {
        if !replaces_injected(cnum) {
            continue;
        }
        if *slot != Linkage::NotLinked {
            return;
        }
    }
    if let Some(injected) = injected {
        assert_eq!(list[injected], Linkage::NotLinked);
        list[injected] = Linkage::Static;
    }
}

/// After the linkage for a crate has been determined we need to verify that
/// there's only going to be one panic runtime in the output.
fn verify_ok(tcx: TyCtxt<'_>, list: &DependencyList) {
    let sess = &tcx.sess;
    if list.is_empty() {
        return;
    }
    let mut panic_runtime = None;
    for (cnum, linkage) in list.iter_enumerated() {
        if let Linkage::NotLinked = *linkage {
            continue;
        }

        if tcx.is_panic_runtime(cnum) {
            if let Some((prev, _)) = panic_runtime {
                let prev_name = tcx.crate_name(prev);
                let cur_name = tcx.crate_name(cnum);
                sess.dcx().emit_err(TwoPanicRuntimes { prev_name, cur_name });
            }
            panic_runtime = Some((
                cnum,
                tcx.required_panic_strategy(cnum).unwrap_or_else(|| {
                    bug!("cannot determine panic strategy of a panic runtime");
                }),
            ));
        }
    }

    // If we found a panic runtime, then we know by this point that it's the
    // only one, but we perform validation here that all the panic strategy
    // compilation modes for the whole DAG are valid.
    if let Some((runtime_cnum, found_strategy)) = panic_runtime {
        let desired_strategy = sess.panic_strategy();

        // First up, validate that our selected panic runtime is indeed exactly
        // our same strategy.
        if found_strategy != desired_strategy {
            sess.dcx().emit_err(BadPanicStrategy {
                runtime: tcx.crate_name(runtime_cnum),
                strategy: desired_strategy,
            });
        }

        // Next up, verify that all other crates are compatible with this panic
        // strategy. If the dep isn't linked, we ignore it, and if our strategy
        // is abort then it's compatible with everything. Otherwise all crates'
        // panic strategy must match our own.
        for (cnum, linkage) in list.iter_enumerated() {
            if let Linkage::NotLinked = *linkage {
                continue;
            }
            if cnum == runtime_cnum || tcx.is_compiler_builtins(cnum) {
                continue;
            }

            if let Some(found_strategy) = tcx.required_panic_strategy(cnum)
                && desired_strategy != found_strategy
            {
                sess.dcx().emit_err(RequiredPanicStrategy {
                    crate_name: tcx.crate_name(cnum),
                    found_strategy,
                    desired_strategy,
                });
            }

            // panic_in_drop_strategy isn't allowed for LOCAL_CRATE
            if cnum != LOCAL_CRATE {
                let found_drop_strategy = tcx.panic_in_drop_strategy(cnum);
                if tcx.sess.opts.unstable_opts.panic_in_drop != found_drop_strategy {
                    sess.dcx().emit_err(IncompatiblePanicInDropStrategy {
                        crate_name: tcx.crate_name(cnum),
                        found_strategy: found_drop_strategy,
                        desired_strategy: tcx.sess.opts.unstable_opts.panic_in_drop,
                    });
                }
            }
        }
    }
}
