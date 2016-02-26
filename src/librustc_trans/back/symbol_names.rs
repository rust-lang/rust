// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! The Rust Linkage Model and Symbol Names
//! =======================================
//!
//! The semantic model of Rust linkage is, broadly, that "there's no global
//! namespace" between crates. Our aim is to preserve the illusion of this
//! model despite the fact that it's not *quite* possible to implement on
//! modern linkers. We initially didn't use system linkers at all, but have
//! been convinced of their utility.
//!
//! There are a few issues to handle:
//!
//!  - Linkers operate on a flat namespace, so we have to flatten names.
//!    We do this using the C++ namespace-mangling technique. Foo::bar
//!    symbols and such.
//!
//!  - Symbols for distinct items with the same *name* need to get different
//!    linkage-names. Examples of this are monomorphizations of functions or
//!    items within anonymous scopes that end up having the same path.
//!
//!  - Symbols in different crates but with same names "within" the crate need
//!    to get different linkage-names.
//!
//!  - Symbol names should be deterministic: Two consecutive runs of the
//!    compiler over the same code base should produce the same symbol names for
//!    the same items.
//!
//!  - Symbol names should not depend on any global properties of the code base,
//!    so that small modifications to the code base do not result in all symbols
//!    changing. In previous versions of the compiler, symbol names incorporated
//!    the SVH (Stable Version Hash) of the crate. This scheme turned out to be
//!    infeasible when used in conjunction with incremental compilation because
//!    small code changes would invalidate all symbols generated previously.
//!
//!  - Even symbols from different versions of the same crate should be able to
//!    live next to each other without conflict.
//!
//! In order to fulfill the above requirements the following scheme is used by
//! the compiler:
//!
//! The main tool for avoiding naming conflicts is the incorporation of a 64-bit
//! hash value into every exported symbol name. Anything that makes a difference
//! to the symbol being named, but does not show up in the regular path needs to
//! be fed into this hash:
//!
//! - Different monomorphizations of the same item have the same path but differ
//!   in their concrete type parameters, so these parameters are part of the
//!   data being digested for the symbol hash.
//!
//! - Rust allows items to be defined in anonymous scopes, such as in
//!   `fn foo() { { fn bar() {} } { fn bar() {} } }`. Both `bar` functions have
//!   the path `foo::bar`, since the anonymous scopes do not contribute to the
//!   path of an item. The compiler already handles this case via so-called
//!   disambiguating `DefPaths` which use indices to distinguish items with the
//!   same name. The DefPaths of the functions above are thus `foo[0]::bar[0]`
//!   and `foo[0]::bar[1]`. In order to incorporate this disambiguation
//!   information into the symbol name too, these indices are fed into the
//!   symbol hash, so that the above two symbols would end up with different
//!   hash values.
//!
//! The two measures described above suffice to avoid intra-crate conflicts. In
//! order to also avoid inter-crate conflicts two more measures are taken:
//!
//! - The name of the crate containing the symbol is prepended to the symbol
//!   name, i.e. symbols are "crate qualified". For example, a function `foo` in
//!   module `bar` in crate `baz` would get a symbol name like
//!   `baz::bar::foo::{hash}` instead of just `bar::foo::{hash}`. This avoids
//!   simple conflicts between functions from different crates.
//!
//! - In order to be able to also use symbols from two versions of the same
//!   crate (which naturally also have the same name), a stronger measure is
//!   required: The compiler accepts an arbitrary "disambiguator" value via the
//!   `-C metadata` commandline argument. This disambiguator is then fed into
//!   the symbol hash of every exported item. Consequently, the symbols in two
//!   identical crates but with different disambiguators are not in conflict
//!   with each other. This facility is mainly intended to be used by build
//!   tools like Cargo.
//!
//! A note on symbol name stability
//! -------------------------------
//! Previous versions of the compiler resorted to feeding NodeIds into the
//! symbol hash in order to disambiguate between items with the same path. The
//! current version of the name generation algorithm takes great care not to do
//! that, since NodeIds are notoriously unstable: A small change to the
//! code base will offset all NodeIds after the change and thus, much as using
//! the SVH in the hash, invalidate an unbounded number of symbol names. This
//! makes re-using previously compiled code for incremental compilation
//! virtually impossible. Thus, symbol hash generation exclusively relies on
//! DefPaths which are much more robust in the face of changes to the code base.

use trans::{CrateContext, Instance, gensym_name};
use util::sha2::{Digest, Sha256};

use rustc::middle::cstore;
use rustc::middle::def_id::DefId;
use rustc::middle::ty::{self, TypeFoldable};
use rustc::front::map::definitions::DefPath;

use std::fmt::Write;
use syntax::ast;
use syntax::parse::token;
use serialize::hex::ToHex;
use super::link;

pub fn def_id_to_string<'tcx>(tcx: &ty::TyCtxt<'tcx>, def_id: DefId) -> String {

    let def_path = tcx.def_path(def_id);
    let mut s = String::with_capacity(def_path.len() * 16);

    let def_path = if def_id.is_local() {
        s.push_str(&tcx.crate_name[..]);
        s.push_str("/");
        s.push_str(&tcx.sess.crate_disambiguator.borrow()[..]);
        &def_path[..]
    } else {
        s.push_str(&tcx.sess.cstore.crate_name(def_id.krate)[..]);
        s.push_str("/");
        s.push_str(&tcx.sess.cstore.crate_disambiguator(def_id.krate));
        &def_path[1..]
    };

    for component in def_path {
        write!(s,
               "::{}[{}]",
               component.data.as_interned_str(),
               component.disambiguator)
            .unwrap();
    }

    s
}

fn get_symbol_hash<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                             def_path: &DefPath,
                             originating_crate: ast::CrateNum,
                             parameters: &[ty::Ty<'tcx>])
                             -> String {
    let tcx = ccx.tcx();

    let mut hash_state = ccx.symbol_hasher().borrow_mut();

    hash_state.reset();

    if originating_crate == cstore::LOCAL_CRATE {
        hash_state.input_str(&tcx.sess.crate_disambiguator.borrow()[..]);
    } else {
        hash_state.input_str(&tcx.sess.cstore.crate_disambiguator(originating_crate));
    }

    for component in def_path {
        let disambiguator_bytes = [(component.disambiguator >>  0) as u8,
                                   (component.disambiguator >>  8) as u8,
                                   (component.disambiguator >> 16) as u8,
                                   (component.disambiguator >> 24) as u8];
        hash_state.input(&disambiguator_bytes);
    }

    for t in parameters {
       assert!(!t.has_erasable_regions());
       assert!(!t.needs_subst());
       let encoded_type = tcx.sess.cstore.encode_type(tcx, t, def_id_to_string);
       hash_state.input(&encoded_type[..]);
    }

    return format!("h{}", truncated_hash_result(&mut *hash_state));

    fn truncated_hash_result(symbol_hasher: &mut Sha256) -> String {
        let output = symbol_hasher.result_bytes();
        // 64 bits should be enough to avoid collisions.
        output[.. 8].to_hex()
    }
}

fn exported_name_with_opt_suffix<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                                           instance: &Instance<'tcx>,
                                           suffix: Option<&str>)
                                           -> String {
    let &Instance { def: mut def_id, params: parameters } = instance;

    if let Some(node_id) = ccx.tcx().map.as_local_node_id(def_id) {
        if let Some(&src_def_id) = ccx.external_srcs().borrow().get(&node_id) {
            def_id = src_def_id;
        }
    }

    let def_path = ccx.tcx().def_path(def_id);
    let hash = get_symbol_hash(ccx, &def_path, def_id.krate, parameters.as_slice());

    let mut path = Vec::with_capacity(16);

    if def_id.is_local() {
        path.push(ccx.tcx().crate_name.clone());
    }

    path.extend(def_path.into_iter().map(|e| e.data.as_interned_str()));

    if let Some(suffix) = suffix {
        path.push(token::intern_and_get_ident(suffix));
    }

    link::mangle(path.into_iter(), Some(&hash[..]))
}

pub fn exported_name<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                               instance: &Instance<'tcx>)
                               -> String {
    exported_name_with_opt_suffix(ccx, instance, None)
}

pub fn exported_name_with_suffix<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                                           instance: &Instance<'tcx>,
                                           suffix: &str)
                                           -> String {
   exported_name_with_opt_suffix(ccx, instance, Some(suffix))
}

/// Only symbols that are invisible outside their compilation unit should use a
/// name generated by this function.
pub fn internal_name_from_type_and_suffix<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                                                    t: ty::Ty<'tcx>,
                                                    suffix: &str)
                                                    -> String {
    let path = [token::intern(&t.to_string()).as_str(),
                gensym_name(suffix).as_str()];
    let hash = get_symbol_hash(ccx, &Vec::new(), cstore::LOCAL_CRATE, &[t]);
    link::mangle(path.iter().cloned(), Some(&hash[..]))
}
