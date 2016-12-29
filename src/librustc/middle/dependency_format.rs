// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

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
//!     1. Each library must appear exactly once in the output.
//!     2. Each rlib contains only one library (it's just an object file)
//!     3. Each dylib can contain more than one library (due to static linking),
//!        and can also bring in many dynamic dependencies.
//!
//! With these constraints in mind, it's generally a very difficult problem to
//! find a solution that's not "all rlibs" or "all dylibs". I have suspicions
//! that NP-ness may come into the picture here...
//!
//! The current selection algorithm below looks mostly similar to:
//!
//!     1. If static linking is required, then require all upstream dependencies
//!        to be available as rlibs. If not, generate an error.
//!     2. If static linking is requested (generating an executable), then
//!        attempt to use all upstream dependencies as rlibs. If any are not
//!        found, bail out and continue to step 3.
//!     3. Static linking has failed, at least one library must be dynamically
//!        linked. Apply a heuristic by greedily maximizing the number of
//!        dynamically linked libraries.
//!     4. Each upstream dependency available as a dynamic library is
//!        registered. The dependencies all propagate, adding to a map. It is
//!        possible for a dylib to add a static library as a dependency, but it
//!        is illegal for two dylibs to add the same static library as a
//!        dependency. The same dylib can be added twice. Additionally, it is
//!        illegal to add a static dependency when it was previously found as a
//!        dylib (and vice versa)
//!     5. After all dynamic dependencies have been traversed, re-traverse the
//!        remaining dependencies and add them statically (if they haven't been
//!        added already).
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

use hir::def_id::CrateNum;

use session;
use session::config;
use middle::cstore::DepKind;
use middle::cstore::LinkagePreference::{self, RequireStatic, RequireDynamic};
use util::nodemap::FxHashMap;
use rustc_back::PanicStrategy;

/// A list of dependencies for a certain crate type.
///
/// The length of this vector is the same as the number of external crates used.
/// The value is None if the crate does not need to be linked (it was found
/// statically in another dylib), or Some(kind) if it needs to be linked as
/// `kind` (either static or dynamic).
pub type DependencyList = Vec<Linkage>;

/// A mapping of all required dependencies for a particular flavor of output.
///
/// This is local to the tcx, and is generally relevant to one session.
pub type Dependencies = FxHashMap<config::CrateType, DependencyList>;

#[derive(Copy, Clone, PartialEq, Debug)]
pub enum Linkage {
    NotLinked,
    IncludedFromDylib,
    Static,
    Dynamic,
}

pub fn calculate(sess: &session::Session) {
    let mut fmts = sess.dependency_formats.borrow_mut();
    for &ty in sess.crate_types.borrow().iter() {
        let linkage = calculate_type(sess, ty);
        verify_ok(sess, &linkage);
        fmts.insert(ty, linkage);
    }
    sess.abort_if_errors();
}

fn calculate_type(sess: &session::Session,
                  ty: config::CrateType) -> DependencyList {
    if !sess.opts.output_types.should_trans() {
        return Vec::new();
    }

    match ty {
        // If the global prefer_dynamic switch is turned off, first attempt
        // static linkage (this can fail).
        config::CrateTypeExecutable if !sess.opts.cg.prefer_dynamic => {
            if let Some(v) = attempt_static(sess) {
                return v;
            }
        }

        // No linkage happens with rlibs, we just needed the metadata (which we
        // got long ago), so don't bother with anything.
        config::CrateTypeRlib => return Vec::new(),

        // Staticlibs and cdylibs must have all static dependencies. If any fail
        // to be found, we generate some nice pretty errors.
        config::CrateTypeStaticlib |
        config::CrateTypeCdylib => {
            if let Some(v) = attempt_static(sess) {
                return v;
            }
            for cnum in sess.cstore.crates() {
                if sess.cstore.dep_kind(cnum).macros_only() { continue }
                let src = sess.cstore.used_crate_source(cnum);
                if src.rlib.is_some() { continue }
                sess.err(&format!("dependency `{}` not found in rlib format",
                                  sess.cstore.crate_name(cnum)));
            }
            return Vec::new();
        }

        // Generating a dylib without `-C prefer-dynamic` means that we're going
        // to try to eagerly statically link all dependencies. This is normally
        // done for end-product dylibs, not intermediate products.
        config::CrateTypeDylib if !sess.opts.cg.prefer_dynamic => {
            if let Some(v) = attempt_static(sess) {
                return v;
            }
        }

        // Everything else falls through below. This will happen either with the
        // `-C prefer-dynamic` or because we're a proc-macro crate. Note that
        // proc-macro crates are required to be dylibs, and they're currently
        // required to link to libsyntax as well.
        config::CrateTypeExecutable |
        config::CrateTypeDylib |
        config::CrateTypeProcMacro => {},
    }

    let mut formats = FxHashMap();

    // Sweep all crates for found dylibs. Add all dylibs, as well as their
    // dependencies, ensuring there are no conflicts. The only valid case for a
    // dependency to be relied upon twice is for both cases to rely on a dylib.
    for cnum in sess.cstore.crates() {
        if sess.cstore.dep_kind(cnum).macros_only() { continue }
        let name = sess.cstore.crate_name(cnum);
        let src = sess.cstore.used_crate_source(cnum);
        if src.dylib.is_some() {
            info!("adding dylib: {}", name);
            add_library(sess, cnum, RequireDynamic, &mut formats);
            let deps = sess.cstore.dylib_dependency_formats(cnum);
            for &(depnum, style) in &deps {
                info!("adding {:?}: {}", style,
                      sess.cstore.crate_name(depnum));
                add_library(sess, depnum, style, &mut formats);
            }
        }
    }

    // Collect what we've got so far in the return vector.
    let last_crate = sess.cstore.crates().len();
    let mut ret = (1..last_crate+1).map(|cnum| {
        match formats.get(&CrateNum::new(cnum)) {
            Some(&RequireDynamic) => Linkage::Dynamic,
            Some(&RequireStatic) => Linkage::IncludedFromDylib,
            None => Linkage::NotLinked,
        }
    }).collect::<Vec<_>>();

    // Run through the dependency list again, and add any missing libraries as
    // static libraries.
    //
    // If the crate hasn't been included yet and it's not actually required
    // (e.g. it's an allocator) then we skip it here as well.
    for cnum in sess.cstore.crates() {
        let src = sess.cstore.used_crate_source(cnum);
        if src.dylib.is_none() &&
           !formats.contains_key(&cnum) &&
           sess.cstore.dep_kind(cnum) == DepKind::Explicit {
            assert!(src.rlib.is_some() || src.rmeta.is_some());
            info!("adding staticlib: {}", sess.cstore.crate_name(cnum));
            add_library(sess, cnum, RequireStatic, &mut formats);
            ret[cnum.as_usize() - 1] = Linkage::Static;
        }
    }

    // We've gotten this far because we're emitting some form of a final
    // artifact which means that we may need to inject dependencies of some
    // form.
    //
    // Things like allocators and panic runtimes may not have been activated
    // quite yet, so do so here.
    activate_injected_dep(sess.injected_allocator.get(), &mut ret,
                          &|cnum| sess.cstore.is_allocator(cnum));
    activate_injected_dep(sess.injected_panic_runtime.get(), &mut ret,
                          &|cnum| sess.cstore.is_panic_runtime(cnum));

    // When dylib B links to dylib A, then when using B we must also link to A.
    // It could be the case, however, that the rlib for A is present (hence we
    // found metadata), but the dylib for A has since been removed.
    //
    // For situations like this, we perform one last pass over the dependencies,
    // making sure that everything is available in the requested format.
    for (cnum, kind) in ret.iter().enumerate() {
        let cnum = CrateNum::new(cnum + 1);
        let src = sess.cstore.used_crate_source(cnum);
        match *kind {
            Linkage::NotLinked |
            Linkage::IncludedFromDylib => {}
            Linkage::Static if src.rlib.is_some() => continue,
            Linkage::Dynamic if src.dylib.is_some() => continue,
            kind => {
                let kind = match kind {
                    Linkage::Static => "rlib",
                    _ => "dylib",
                };
                let name = sess.cstore.crate_name(cnum);
                sess.err(&format!("crate `{}` required to be available in {}, \
                                  but it was not available in this form",
                                  name, kind));
            }
        }
    }

    return ret;
}

fn add_library(sess: &session::Session,
               cnum: CrateNum,
               link: LinkagePreference,
               m: &mut FxHashMap<CrateNum, LinkagePreference>) {
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
                sess.struct_err(&format!("cannot satisfy dependencies so `{}` only \
                                          shows up once", sess.cstore.crate_name(cnum)))
                    .help("having upstream crates all available in one format \
                           will likely make this go away")
                    .emit();
            }
        }
        None => { m.insert(cnum, link); }
    }
}

fn attempt_static(sess: &session::Session) -> Option<DependencyList> {
    let crates = sess.cstore.used_crates(RequireStatic);
    if !crates.iter().by_ref().all(|&(_, ref p)| p.is_some()) {
        return None
    }

    // All crates are available in an rlib format, so we're just going to link
    // everything in explicitly so long as it's actually required.
    let last_crate = sess.cstore.crates().len();
    let mut ret = (1..last_crate+1).map(|cnum| {
        if sess.cstore.dep_kind(CrateNum::new(cnum)) == DepKind::Explicit {
            Linkage::Static
        } else {
            Linkage::NotLinked
        }
    }).collect::<Vec<_>>();

    // Our allocator/panic runtime may not have been linked above if it wasn't
    // explicitly linked, which is the case for any injected dependency. Handle
    // that here and activate them.
    activate_injected_dep(sess.injected_allocator.get(), &mut ret,
                          &|cnum| sess.cstore.is_allocator(cnum));
    activate_injected_dep(sess.injected_panic_runtime.get(), &mut ret,
                          &|cnum| sess.cstore.is_panic_runtime(cnum));

    Some(ret)
}

// Given a list of how to link upstream dependencies so far, ensure that an
// injected dependency is activated. This will not do anything if one was
// transitively included already (e.g. via a dylib or explicitly so).
//
// If an injected dependency was not found then we're guaranteed the
// metadata::creader module has injected that dependency (not listed as
// a required dependency) in one of the session's field. If this field is not
// set then this compilation doesn't actually need the dependency and we can
// also skip this step entirely.
fn activate_injected_dep(injected: Option<CrateNum>,
                         list: &mut DependencyList,
                         replaces_injected: &Fn(CrateNum) -> bool) {
    for (i, slot) in list.iter().enumerate() {
        let cnum = CrateNum::new(i + 1);
        if !replaces_injected(cnum) {
            continue
        }
        if *slot != Linkage::NotLinked {
            return
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
fn verify_ok(sess: &session::Session, list: &[Linkage]) {
    if list.len() == 0 {
        return
    }
    let mut allocator = None;
    let mut panic_runtime = None;
    for (i, linkage) in list.iter().enumerate() {
        if let Linkage::NotLinked = *linkage {
            continue
        }
        let cnum = CrateNum::new(i + 1);
        if sess.cstore.is_allocator(cnum) {
            if let Some(prev) = allocator {
                let prev_name = sess.cstore.crate_name(prev);
                let cur_name = sess.cstore.crate_name(cnum);
                sess.err(&format!("cannot link together two \
                                   allocators: {} and {}",
                                  prev_name, cur_name));
            }
            allocator = Some(cnum);
        }

        if sess.cstore.is_panic_runtime(cnum) {
            if let Some((prev, _)) = panic_runtime {
                let prev_name = sess.cstore.crate_name(prev);
                let cur_name = sess.cstore.crate_name(cnum);
                sess.err(&format!("cannot link together two \
                                   panic runtimes: {} and {}",
                                  prev_name, cur_name));
            }
            panic_runtime = Some((cnum, sess.cstore.panic_strategy(cnum)));
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
            sess.err(&format!("the linked panic runtime `{}` is \
                               not compiled with this crate's \
                               panic strategy `{}`",
                              sess.cstore.crate_name(cnum),
                              desired_strategy.desc()));
        }

        // Next up, verify that all other crates are compatible with this panic
        // strategy. If the dep isn't linked, we ignore it, and if our strategy
        // is abort then it's compatible with everything. Otherwise all crates'
        // panic strategy must match our own.
        for (i, linkage) in list.iter().enumerate() {
            if let Linkage::NotLinked = *linkage {
                continue
            }
            if desired_strategy == PanicStrategy::Abort {
                continue
            }
            let cnum = CrateNum::new(i + 1);
            let found_strategy = sess.cstore.panic_strategy(cnum);
            if desired_strategy == found_strategy {
                continue
            }

            sess.err(&format!("the crate `{}` is compiled with the \
                               panic strategy `{}` which is \
                               incompatible with this crate's \
                               strategy of `{}`",
                              sess.cstore.crate_name(cnum),
                              found_strategy.desc(),
                              desired_strategy.desc()));
        }
    }
}
