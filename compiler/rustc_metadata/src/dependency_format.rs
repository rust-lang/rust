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

use crate::creader::CStore;

use rustc_data_structures::fx::FxHashMap;
use rustc_hir::def_id::CrateNum;
use rustc_middle::middle::cstore::CrateDepKind;
use rustc_middle::middle::cstore::LinkagePreference::{self, RequireDynamic, RequireStatic};
use rustc_middle::middle::dependency_format::{Dependencies, DependencyList, Linkage};
use rustc_middle::ty::TyCtxt;
use rustc_session::config::CrateType;
use rustc_target::spec::PanicStrategy;

crate fn calculate(tcx: TyCtxt<'_>) -> Dependencies {
    tcx.sess
        .crate_types()
        .iter()
        .map(|&ty| {
            let linkage = calculate_type(tcx, ty);
            verify_ok(tcx, &linkage);
            (ty, linkage)
        })
        .collect::<Vec<_>>()
}

fn calculate_type(tcx: TyCtxt<'_>, ty: CrateType) -> DependencyList {
    let sess = &tcx.sess;

    if !sess.opts.output_types.should_codegen() {
        return Vec::new();
    }

    let preferred_linkage = match ty {
        // Generating a dylib without `-C prefer-dynamic` means that we're going
        // to try to eagerly statically link all dependencies. This is normally
        // done for end-product dylibs, not intermediate products.
        //
        // Treat cdylibs similarly. If `-C prefer-dynamic` is set, the caller may
        // be code-size conscious, but without it, it makes sense to statically
        // link a cdylib.
        CrateType::Dylib | CrateType::Cdylib if !sess.opts.cg.prefer_dynamic => Linkage::Static,
        CrateType::Dylib | CrateType::Cdylib => Linkage::Dynamic,

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

        // staticlibs must have all static dependencies.
        CrateType::Staticlib => Linkage::Static,
    };

    if preferred_linkage == Linkage::NotLinked {
        // If the crate is not linked, there are no link-time dependencies.
        return Vec::new();
    }

    if preferred_linkage == Linkage::Static {
        // Attempt static linkage first. For dylibs and executables, we may be
        // able to retry below with dynamic linkage.
        if let Some(v) = attempt_static(tcx) {
            return v;
        }

        // Staticlibs and static executables must have all static dependencies.
        // If any are not found, generate some nice pretty errors.
        if ty == CrateType::Staticlib
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
                sess.err(&format!(
                    "crate `{}` required to be available in rlib format, \
                                   but was not found in this form",
                    tcx.crate_name(cnum)
                ));
            }
            return Vec::new();
        }
    }

    let mut formats = FxHashMap::default();

    // Sweep all crates for found dylibs. Add all dylibs, as well as their
    // dependencies, ensuring there are no conflicts. The only valid case for a
    // dependency to be relied upon twice is for both cases to rely on a dylib.
    for &cnum in tcx.crates(()).iter() {
        if tcx.dep_kind(cnum).macros_only() {
            continue;
        }
        let name = tcx.crate_name(cnum);
        let src = tcx.used_crate_source(cnum);
        if src.dylib.is_some() {
            tracing::info!("adding dylib: {}", name);
            add_library(tcx, cnum, RequireDynamic, &mut formats);
            let deps = tcx.dylib_dependency_formats(cnum);
            for &(depnum, style) in deps.iter() {
                tracing::info!("adding {:?}: {}", style, tcx.crate_name(depnum));
                add_library(tcx, depnum, style, &mut formats);
            }
        }
    }

    // Collect what we've got so far in the return vector.
    let last_crate = tcx.crates(()).len();
    let mut ret = (1..last_crate + 1)
        .map(|cnum| match formats.get(&CrateNum::new(cnum)) {
            Some(&RequireDynamic) => Linkage::Dynamic,
            Some(&RequireStatic) => Linkage::IncludedFromDylib,
            None => Linkage::NotLinked,
        })
        .collect::<Vec<_>>();

    // Run through the dependency list again, and add any missing libraries as
    // static libraries.
    //
    // If the crate hasn't been included yet and it's not actually required
    // (e.g., it's an allocator) then we skip it here as well.
    for &cnum in tcx.crates(()).iter() {
        let src = tcx.used_crate_source(cnum);
        if src.dylib.is_none()
            && !formats.contains_key(&cnum)
            && tcx.dep_kind(cnum) == CrateDepKind::Explicit
        {
            assert!(src.rlib.is_some() || src.rmeta.is_some());
            tracing::info!("adding staticlib: {}", tcx.crate_name(cnum));
            add_library(tcx, cnum, RequireStatic, &mut formats);
            ret[cnum.as_usize() - 1] = Linkage::Static;
        }
    }

    // We've gotten this far because we're emitting some form of a final
    // artifact which means that we may need to inject dependencies of some
    // form.
    //
    // Things like allocators and panic runtimes may not have been activated
    // quite yet, so do so here.
    activate_injected_dep(CStore::from_tcx(tcx).injected_panic_runtime(), &mut ret, &|cnum| {
        tcx.is_panic_runtime(cnum)
    });

    // When dylib B links to dylib A, then when using B we must also link to A.
    // It could be the case, however, that the rlib for A is present (hence we
    // found metadata), but the dylib for A has since been removed.
    //
    // For situations like this, we perform one last pass over the dependencies,
    // making sure that everything is available in the requested format.
    for (cnum, kind) in ret.iter().enumerate() {
        let cnum = CrateNum::new(cnum + 1);
        let src = tcx.used_crate_source(cnum);
        match *kind {
            Linkage::NotLinked | Linkage::IncludedFromDylib => {}
            Linkage::Static if src.rlib.is_some() => continue,
            Linkage::Dynamic if src.dylib.is_some() => continue,
            kind => {
                let kind = match kind {
                    Linkage::Static => "rlib",
                    _ => "dylib",
                };
                sess.err(&format!(
                    "crate `{}` required to be available in {} format, \
                                   but was not found in this form",
                    tcx.crate_name(cnum),
                    kind
                ));
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
                tcx.sess
                    .struct_err(&format!(
                        "cannot satisfy dependencies so `{}` only \
                                              shows up once",
                        tcx.crate_name(cnum)
                    ))
                    .help(
                        "having upstream crates all available in one format \
                           will likely make this go away",
                    )
                    .emit();
            }
        }
        None => {
            m.insert(cnum, link);
        }
    }
}

fn attempt_static(tcx: TyCtxt<'_>) -> Option<DependencyList> {
    let all_crates_available_as_rlib = tcx
        .crates(())
        .iter()
        .copied()
        .filter_map(|cnum| {
            if tcx.dep_kind(cnum).macros_only() {
                return None;
            }
            Some(tcx.used_crate_source(cnum).rlib.is_some())
        })
        .all(|is_rlib| is_rlib);
    if !all_crates_available_as_rlib {
        return None;
    }

    // All crates are available in an rlib format, so we're just going to link
    // everything in explicitly so long as it's actually required.
    let mut ret = tcx
        .crates(())
        .iter()
        .map(|&cnum| {
            if tcx.dep_kind(cnum) == CrateDepKind::Explicit {
                Linkage::Static
            } else {
                Linkage::NotLinked
            }
        })
        .collect::<Vec<_>>();

    // Our allocator/panic runtime may not have been linked above if it wasn't
    // explicitly linked, which is the case for any injected dependency. Handle
    // that here and activate them.
    activate_injected_dep(CStore::from_tcx(tcx).injected_panic_runtime(), &mut ret, &|cnum| {
        tcx.is_panic_runtime(cnum)
    });

    Some(ret)
}

// Given a list of how to link upstream dependencies so far, ensure that an
// injected dependency is activated. This will not do anything if one was
// transitively included already (e.g., via a dylib or explicitly so).
//
// If an injected dependency was not found then we're guaranteed the
// metadata::creader module has injected that dependency (not listed as
// a required dependency) in one of the session's field. If this field is not
// set then this compilation doesn't actually need the dependency and we can
// also skip this step entirely.
fn activate_injected_dep(
    injected: Option<CrateNum>,
    list: &mut DependencyList,
    replaces_injected: &dyn Fn(CrateNum) -> bool,
) {
    for (i, slot) in list.iter().enumerate() {
        let cnum = CrateNum::new(i + 1);
        if !replaces_injected(cnum) {
            continue;
        }
        if *slot != Linkage::NotLinked {
            return;
        }
    }
    if let Some(injected) = injected {
        let idx = injected.as_usize() - 1;
        assert_eq!(list[idx], Linkage::NotLinked);
        list[idx] = Linkage::Static;
    }
}

// After the linkage for a crate has been determined we need to verify that
// there's only going to be one allocator in the output.
fn verify_ok(tcx: TyCtxt<'_>, list: &[Linkage]) {
    let sess = &tcx.sess;
    if list.is_empty() {
        return;
    }
    let mut panic_runtime = None;
    for (i, linkage) in list.iter().enumerate() {
        if let Linkage::NotLinked = *linkage {
            continue;
        }
        let cnum = CrateNum::new(i + 1);

        if tcx.is_panic_runtime(cnum) {
            if let Some((prev, _)) = panic_runtime {
                let prev_name = tcx.crate_name(prev);
                let cur_name = tcx.crate_name(cnum);
                sess.err(&format!(
                    "cannot link together two \
                                   panic runtimes: {} and {}",
                    prev_name, cur_name
                ));
            }
            panic_runtime = Some((cnum, tcx.panic_strategy(cnum)));
        }
    }

    // If we found a panic runtime, then we know by this point that it's the
    // only one, but we perform validation here that all the panic strategy
    // compilation modes for the whole DAG are valid.
    if let Some((cnum, found_strategy)) = panic_runtime {
        let desired_strategy = sess.panic_strategy();

        // First up, validate that our selected panic runtime is indeed exactly
        // our same strategy.
        if found_strategy != desired_strategy {
            sess.err(&format!(
                "the linked panic runtime `{}` is \
                               not compiled with this crate's \
                               panic strategy `{}`",
                tcx.crate_name(cnum),
                desired_strategy.desc()
            ));
        }

        // Next up, verify that all other crates are compatible with this panic
        // strategy. If the dep isn't linked, we ignore it, and if our strategy
        // is abort then it's compatible with everything. Otherwise all crates'
        // panic strategy must match our own.
        for (i, linkage) in list.iter().enumerate() {
            if let Linkage::NotLinked = *linkage {
                continue;
            }
            if desired_strategy == PanicStrategy::Abort {
                continue;
            }
            let cnum = CrateNum::new(i + 1);
            if tcx.is_compiler_builtins(cnum) {
                continue;
            }

            let found_strategy = tcx.panic_strategy(cnum);
            if desired_strategy != found_strategy {
                sess.err(&format!(
                    "the crate `{}` is compiled with the \
                               panic strategy `{}` which is \
                               incompatible with this crate's \
                               strategy of `{}`",
                    tcx.crate_name(cnum),
                    found_strategy.desc(),
                    desired_strategy.desc()
                ));
            }

            let found_drop_strategy = tcx.panic_in_drop_strategy(cnum);
            if tcx.sess.opts.debugging_opts.panic_in_drop != found_drop_strategy {
                sess.err(&format!(
                    "the crate `{}` is compiled with the \
                               panic-in-drop strategy `{}` which is \
                               incompatible with this crate's \
                               strategy of `{}`",
                    tcx.crate_name(cnum),
                    found_drop_strategy.desc(),
                    tcx.sess.opts.debugging_opts.panic_in_drop.desc()
                ));
            }
        }
    }
}
