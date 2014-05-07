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

use collections::HashMap;
use syntax::ast;

use driver::session;
use metadata::cstore;
use metadata::csearch;
use middle::ty;

/// A list of dependencies for a certain crate type.
///
/// The length of this vector is the same as the number of external crates used.
/// The value is None if the crate does not need to be linked (it was found
/// statically in another dylib), or Some(kind) if it needs to be linked as
/// `kind` (either static or dynamic).
pub type DependencyList = Vec<Option<cstore::LinkagePreference>>;

/// A mapping of all required dependencies for a particular flavor of output.
///
/// This is local to the tcx, and is generally relevant to one session.
pub type Dependencies = HashMap<session::CrateType, DependencyList>;

pub fn calculate(tcx: &ty::ctxt) {
    let mut fmts = tcx.dependency_formats.borrow_mut();
    for &ty in tcx.sess.crate_types.borrow().iter() {
        fmts.insert(ty, calculate_type(&tcx.sess, ty));
    }
    tcx.sess.abort_if_errors();
}

fn calculate_type(sess: &session::Session,
                  ty: session::CrateType) -> DependencyList {
    match ty {
        // If the global prefer_dynamic switch is turned off, first attempt
        // static linkage (this can fail).
        session::CrateTypeExecutable if !sess.opts.cg.prefer_dynamic => {
            match attempt_static(sess) {
                Some(v) => return v,
                None => {}
            }
        }

        // No linkage happens with rlibs, we just needed the metadata (which we
        // got long ago), so don't bother with anything.
        session::CrateTypeRlib => return Vec::new(),

        // Staticlibs must have all static dependencies. If any fail to be
        // found, we generate some nice pretty errors.
        session::CrateTypeStaticlib => {
            match attempt_static(sess) {
                Some(v) => return v,
                None => {}
            }
            sess.cstore.iter_crate_data(|cnum, data| {
                let src = sess.cstore.get_used_crate_source(cnum).unwrap();
                if src.rlib.is_some() { return }
                sess.err(format!("dependency `{}` not found in rlib format",
                                 data.name));
            });
            return Vec::new();
        }

        // Everything else falls through below
        session::CrateTypeExecutable | session::CrateTypeDylib => {},
    }

    let mut formats = HashMap::new();

    // Sweep all crates for found dylibs. Add all dylibs, as well as their
    // dependencies, ensuring there are no conflicts. The only valid case for a
    // dependency to be relied upon twice is for both cases to rely on a dylib.
    sess.cstore.iter_crate_data(|cnum, data| {
        let src = sess.cstore.get_used_crate_source(cnum).unwrap();
        if src.dylib.is_some() {
            add_library(sess, cnum, cstore::RequireDynamic, &mut formats);
            debug!("adding dylib: {}", data.name);
            let deps = csearch::get_dylib_dependency_formats(&sess.cstore, cnum);
            for &(depnum, style) in deps.iter() {
                add_library(sess, depnum, style, &mut formats);
                debug!("adding {}: {}", style,
                       sess.cstore.get_crate_data(depnum).name.clone());
            }
        }
    });

    // Collect what we've got so far in the return vector.
    let mut ret = range(1, sess.cstore.next_crate_num()).map(|i| {
        match formats.find(&i).map(|v| *v) {
            v @ Some(cstore::RequireDynamic) => v,
            _ => None,
        }
    }).collect::<Vec<_>>();

    // Run through the dependency list again, and add any missing libraries as
    // static libraries.
    sess.cstore.iter_crate_data(|cnum, data| {
        let src = sess.cstore.get_used_crate_source(cnum).unwrap();
        if src.dylib.is_none() && !formats.contains_key(&cnum) {
            assert!(src.rlib.is_some());
            add_library(sess, cnum, cstore::RequireStatic, &mut formats);
            *ret.get_mut(cnum as uint - 1) = Some(cstore::RequireStatic);
            debug!("adding staticlib: {}", data.name);
        }
    });

    // When dylib B links to dylib A, then when using B we must also link to A.
    // It could be the case, however, that the rlib for A is present (hence we
    // found metadata), but the dylib for A has since been removed.
    //
    // For situations like this, we perform one last pass over the dependencies,
    // making sure that everything is available in the requested format.
    for (cnum, kind) in ret.iter().enumerate() {
        let cnum = cnum as ast::CrateNum;
        let src = sess.cstore.get_used_crate_source(cnum + 1).unwrap();
        match *kind {
            None => continue,
            Some(cstore::RequireStatic) if src.rlib.is_some() => continue,
            Some(cstore::RequireDynamic) if src.dylib.is_some() => continue,
            Some(kind) => {
                let data = sess.cstore.get_crate_data(cnum + 1);
                sess.err(format!("crate `{}` required to be available in {}, \
                                  but it was not available in this form",
                                 data.name,
                                 match kind {
                                     cstore::RequireStatic => "rlib",
                                     cstore::RequireDynamic => "dylib",
                                 }));
            }
        }
    }

    return ret;
}

fn add_library(sess: &session::Session,
               cnum: ast::CrateNum,
               link: cstore::LinkagePreference,
               m: &mut HashMap<ast::CrateNum, cstore::LinkagePreference>) {
    match m.find(&cnum) {
        Some(&link2) => {
            // If the linkages differ, then we'd have two copies of the library
            // if we continued linking. If the linkages are both static, then we
            // would also have two copies of the library (static from two
            // different locations).
            //
            // This error is probably a little obscure, but I imagine that it
            // can be refined over time.
            if link2 != link || link == cstore::RequireStatic {
                let data = sess.cstore.get_crate_data(cnum);
                sess.err(format!("cannot satisfy dependencies so `{}` only \
                                  shows up once", data.name));
                sess.note("having upstream crates all available in one format \
                           will likely make this go away");
            }
        }
        None => { m.insert(cnum, link); }
    }
}

fn attempt_static(sess: &session::Session) -> Option<DependencyList> {
    let crates = sess.cstore.get_used_crates(cstore::RequireStatic);
    if crates.iter().all(|&(_, ref p)| p.is_some()) {
        Some(crates.move_iter().map(|_| Some(cstore::RequireStatic)).collect())
    } else {
        None
    }
}
