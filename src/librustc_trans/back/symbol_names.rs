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

use common::{CrateContext, SharedCrateContext, gensym_name};
use monomorphize::Instance;
use util::sha2::{Digest, Sha256};

use rustc::middle::{cstore, weak_lang_items};
use rustc::hir::def_id::DefId;
use rustc::hir::map as hir_map;
use rustc::ty::{self, TyCtxt, TypeFoldable};
use rustc::ty::item_path::{self, ItemPathBuffer, RootMode};
use rustc::hir::map::definitions::{DefPath, DefPathData};

use std::fmt::Write;
use syntax::attr;
use syntax::parse::token::{self, InternedString};
use serialize::hex::ToHex;

pub fn def_id_to_string<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>, def_id: DefId) -> String {
    let def_path = tcx.def_path(def_id);
    def_path_to_string(tcx, &def_path)
}

fn def_path_to_string<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>, def_path: &DefPath) -> String {
    let mut s = String::with_capacity(def_path.data.len() * 16);

    s.push_str(&tcx.crate_name(def_path.krate));
    s.push_str("/");
    s.push_str(&tcx.crate_disambiguator(def_path.krate));

    for component in &def_path.data {
        write!(s,
               "::{}[{}]",
               component.data.as_interned_str(),
               component.disambiguator)
            .unwrap();
    }

    s
}

fn get_symbol_hash<'a, 'tcx>(scx: &SharedCrateContext<'a, 'tcx>,

                             // path to the item this name is for
                             def_path: &DefPath,

                             // type of the item, without any generic
                             // parameters substituted; this is
                             // included in the hash as a kind of
                             // safeguard.
                             item_type: ty::Ty<'tcx>,

                             // values for generic type parameters,
                             // if any.
                             parameters: &[ty::Ty<'tcx>])
                             -> String {
    debug!("get_symbol_hash(def_path={:?}, parameters={:?})",
           def_path, parameters);

    let tcx = scx.tcx();

    let mut hash_state = scx.symbol_hasher().borrow_mut();

    hash_state.reset();

    // the main symbol name is not necessarily unique; hash in the
    // compiler's internal def-path, guaranteeing each symbol has a
    // truly unique path
    hash_state.input_str(&def_path_to_string(tcx, def_path));

    // Include the main item-type. Note that, in this case, the
    // assertions about `needs_subst` may not hold, but this item-type
    // ought to be the same for every reference anyway.
    assert!(!item_type.has_erasable_regions());
    let encoded_item_type = tcx.sess.cstore.encode_type(tcx, item_type, def_id_to_string);
    hash_state.input(&encoded_item_type[..]);

    // also include any type parameters (for generic items)
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

impl<'a, 'tcx> Instance<'tcx> {
    pub fn symbol_name(self, scx: &SharedCrateContext<'a, 'tcx>) -> String {
        let Instance { def: def_id, ref substs } = self;

        debug!("symbol_name(def_id={:?}, substs={:?})",
               def_id, substs);

        let node_id = scx.tcx().map.as_local_node_id(def_id);

        if let Some(id) = node_id {
            if scx.sess().plugin_registrar_fn.get() == Some(id) {
                let svh = &scx.link_meta().crate_hash;
                let idx = def_id.index;
                return scx.sess().generate_plugin_registrar_symbol(svh, idx);
            }
        }

        // FIXME(eddyb) Precompute a custom symbol name based on attributes.
        let attrs = scx.tcx().get_attrs(def_id);
        let is_foreign = if let Some(id) = node_id {
            match scx.tcx().map.get(id) {
                hir_map::NodeForeignItem(_) => true,
                _ => false
            }
        } else {
            scx.sess().cstore.is_foreign_item(def_id)
        };

        if let Some(name) = weak_lang_items::link_name(&attrs) {
            return name.to_string();
        }

        if is_foreign {
            if let Some(name) = attr::first_attr_value_str_by_name(&attrs, "link_name") {
                return name.to_string();
            }
            // Don't mangle foreign items.
            return scx.tcx().item_name(def_id).as_str().to_string();
        }

        if let Some(name) = attr::find_export_name_attr(scx.sess().diagnostic(), &attrs) {
            // Use provided name
            return name.to_string();
        }

        if attr::contains_name(&attrs, "no_mangle") {
            // Don't mangle
            return scx.tcx().item_name(def_id).as_str().to_string();
        }

        let def_path = scx.tcx().def_path(def_id);

        // We want to compute the "type" of this item. Unfortunately, some
        // kinds of items (e.g., closures) don't have an entry in the
        // item-type array. So walk back up the find the closest parent
        // that DOES have an entry.
        let mut ty_def_id = def_id;
        let instance_ty;
        loop {
            let key = scx.tcx().def_key(ty_def_id);
            match key.disambiguated_data.data {
                DefPathData::TypeNs(_) |
                DefPathData::ValueNs(_) => {
                    instance_ty = scx.tcx().lookup_item_type(ty_def_id);
                    break;
                }
                _ => {
                    // if we're making a symbol for something, there ought
                    // to be a value or type-def or something in there
                    // *somewhere*
                    ty_def_id.index = key.parent.unwrap_or_else(|| {
                        bug!("finding type for {:?}, encountered def-id {:?} with no \
                             parent", def_id, ty_def_id);
                    });
                }
            }
        }

        // Erase regions because they may not be deterministic when hashed
        // and should not matter anyhow.
        let instance_ty = scx.tcx().erase_regions(&instance_ty.ty);

        let hash = get_symbol_hash(scx, &def_path, instance_ty, substs.types.as_slice());

        let mut buffer = SymbolPathBuffer {
            names: Vec::with_capacity(def_path.data.len())
        };

        item_path::with_forced_absolute_paths(|| {
            scx.tcx().push_item_path(&mut buffer, def_id);
        });

        mangle(buffer.names.into_iter(), Some(&hash[..]))
    }
}

struct SymbolPathBuffer {
    names: Vec<InternedString>,
}

impl ItemPathBuffer for SymbolPathBuffer {
    fn root_mode(&self) -> &RootMode {
        const ABSOLUTE: &'static RootMode = &RootMode::Absolute;
        ABSOLUTE
    }

    fn push(&mut self, text: &str) {
        self.names.push(token::intern(text).as_str());
    }
}

/// Only symbols that are invisible outside their compilation unit should use a
/// name generated by this function.
pub fn internal_name_from_type_and_suffix<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                                                    t: ty::Ty<'tcx>,
                                                    suffix: &str)
                                                    -> String {
    let path = [token::intern(&t.to_string()).as_str(),
                gensym_name(suffix).as_str()];
    let def_path = DefPath {
        data: vec![],
        krate: cstore::LOCAL_CRATE,
    };
    let hash = get_symbol_hash(ccx.shared(), &def_path, t, &[]);
    mangle(path.iter().cloned(), Some(&hash[..]))
}

// Name sanitation. LLVM will happily accept identifiers with weird names, but
// gas doesn't!
// gas accepts the following characters in symbols: a-z, A-Z, 0-9, ., _, $
pub fn sanitize(s: &str) -> String {
    let mut result = String::new();
    for c in s.chars() {
        match c {
            // Escape these with $ sequences
            '@' => result.push_str("$SP$"),
            '*' => result.push_str("$BP$"),
            '&' => result.push_str("$RF$"),
            '<' => result.push_str("$LT$"),
            '>' => result.push_str("$GT$"),
            '(' => result.push_str("$LP$"),
            ')' => result.push_str("$RP$"),
            ',' => result.push_str("$C$"),

            // '.' doesn't occur in types and functions, so reuse it
            // for ':' and '-'
            '-' | ':' => result.push('.'),

            // These are legal symbols
            'a' ... 'z'
            | 'A' ... 'Z'
            | '0' ... '9'
            | '_' | '.' | '$' => result.push(c),

            _ => {
                result.push('$');
                for c in c.escape_unicode().skip(1) {
                    match c {
                        '{' => {},
                        '}' => result.push('$'),
                        c => result.push(c),
                    }
                }
            }
        }
    }

    // Underscore-qualify anything that didn't start as an ident.
    if !result.is_empty() &&
        result.as_bytes()[0] != '_' as u8 &&
        ! (result.as_bytes()[0] as char).is_xid_start() {
        return format!("_{}", &result[..]);
    }

    return result;
}

pub fn mangle<PI: Iterator<Item=InternedString>>(path: PI, hash: Option<&str>) -> String {
    // Follow C++ namespace-mangling style, see
    // http://en.wikipedia.org/wiki/Name_mangling for more info.
    //
    // It turns out that on OSX you can actually have arbitrary symbols in
    // function names (at least when given to LLVM), but this is not possible
    // when using unix's linker. Perhaps one day when we just use a linker from LLVM
    // we won't need to do this name mangling. The problem with name mangling is
    // that it seriously limits the available characters. For example we can't
    // have things like &T in symbol names when one would theoretically
    // want them for things like impls of traits on that type.
    //
    // To be able to work on all platforms and get *some* reasonable output, we
    // use C++ name-mangling.

    let mut n = String::from("_ZN"); // _Z == Begin name-sequence, N == nested

    fn push(n: &mut String, s: &str) {
        let sani = sanitize(s);
        n.push_str(&format!("{}{}", sani.len(), sani));
    }

    // First, connect each component with <len, name> pairs.
    for data in path {
        push(&mut n, &data);
    }

    if let Some(s) = hash {
        push(&mut n, s)
    }

    n.push('E'); // End name-sequence.
    n
}
