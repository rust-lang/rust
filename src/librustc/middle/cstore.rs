// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// the rustc crate store interface. This also includes types that
// are *mostly* used as a part of that interface, but these should
// probably get a better home if someone can find one.

use hir::def::{self, Def};
use hir::def_id::{CrateNum, DefId, DefIndex};
use hir::map as hir_map;
use hir::map::definitions::{Definitions, DefKey};
use hir::svh::Svh;
use middle::lang_items;
use ty::{self, Ty, TyCtxt};
use mir::repr::Mir;
use mir::mir_map::MirMap;
use session::Session;
use session::config::PanicStrategy;
use session::search_paths::PathKind;
use util::nodemap::{NodeSet, DefIdMap};
use std::path::PathBuf;
use std::rc::Rc;
use syntax::ast;
use syntax::attr;
use syntax::ext::base::MultiItemModifier;
use syntax::ptr::P;
use syntax::parse::token::InternedString;
use syntax_pos::Span;
use rustc_back::target::Target;
use hir;
use hir::intravisit::Visitor;

pub use self::NativeLibraryKind::{NativeStatic, NativeFramework, NativeUnknown};

// lonely orphan structs and enums looking for a better home

#[derive(Clone, Debug)]
pub struct LinkMeta {
    pub crate_name: String,
    pub crate_hash: Svh,
}

// Where a crate came from on the local filesystem. One of these two options
// must be non-None.
#[derive(PartialEq, Clone, Debug)]
pub struct CrateSource {
    pub dylib: Option<(PathBuf, PathKind)>,
    pub rlib: Option<(PathBuf, PathKind)>,
    pub cnum: CrateNum,
}

#[derive(Copy, Debug, PartialEq, Clone, RustcEncodable, RustcDecodable)]
pub enum LinkagePreference {
    RequireDynamic,
    RequireStatic,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, RustcEncodable, RustcDecodable)]
pub enum NativeLibraryKind {
    NativeStatic,    // native static library (.a archive)
    NativeFramework, // OSX-specific
    NativeUnknown,   // default way to specify a dynamic library
}

/// The data we save and restore about an inlined item or method.  This is not
/// part of the AST that we parse from a file, but it becomes part of the tree
/// that we trans.
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub enum InlinedItem {
    Item(DefId /* def-id in source crate */, P<hir::Item>),
    TraitItem(DefId /* impl id */, P<hir::TraitItem>),
    ImplItem(DefId /* impl id */, P<hir::ImplItem>)
}

/// A borrowed version of `hir::InlinedItem`.
#[derive(Clone, Copy, PartialEq, Eq, RustcEncodable, Hash, Debug)]
pub enum InlinedItemRef<'a> {
    Item(DefId, &'a hir::Item),
    TraitItem(DefId, &'a hir::TraitItem),
    ImplItem(DefId, &'a hir::ImplItem)
}

#[derive(Copy, Clone, Debug)]
pub struct ExternCrate {
    /// def_id of an `extern crate` in the current crate that caused
    /// this crate to be loaded; note that there could be multiple
    /// such ids
    pub def_id: DefId,

    /// span of the extern crate that caused this to be loaded
    pub span: Span,

    /// If true, then this crate is the crate named by the extern
    /// crate referenced above. If false, then this crate is a dep
    /// of the crate.
    pub direct: bool,

    /// Number of links to reach the extern crate `def_id`
    /// declaration; used to select the extern crate with the shortest
    /// path
    pub path_len: usize,
}

/// A store of Rust crates, through with their metadata
/// can be accessed.
pub trait CrateStore<'tcx> {
    // item info
    fn describe_def(&self, def: DefId) -> Option<Def>;
    fn stability(&self, def: DefId) -> Option<attr::Stability>;
    fn deprecation(&self, def: DefId) -> Option<attr::Deprecation>;
    fn visibility(&self, def: DefId) -> ty::Visibility;
    fn closure_kind(&self, def_id: DefId) -> ty::ClosureKind;
    fn closure_ty<'a>(&self, tcx: TyCtxt<'a, 'tcx, 'tcx>, def_id: DefId)
                      -> ty::ClosureTy<'tcx>;
    fn item_variances(&self, def: DefId) -> Vec<ty::Variance>;
    fn item_type<'a>(&self, tcx: TyCtxt<'a, 'tcx, 'tcx>, def: DefId)
                     -> Ty<'tcx>;
    fn visible_parent_map<'a>(&'a self) -> ::std::cell::RefMut<'a, DefIdMap<DefId>>;
    fn item_predicates<'a>(&self, tcx: TyCtxt<'a, 'tcx, 'tcx>, def: DefId)
                           -> ty::GenericPredicates<'tcx>;
    fn item_super_predicates<'a>(&self, tcx: TyCtxt<'a, 'tcx, 'tcx>, def: DefId)
                                 -> ty::GenericPredicates<'tcx>;
    fn item_generics<'a>(&self, tcx: TyCtxt<'a, 'tcx, 'tcx>, def: DefId)
                         -> ty::Generics<'tcx>;
    fn item_attrs(&self, def_id: DefId) -> Vec<ast::Attribute>;
    fn trait_def<'a>(&self, tcx: TyCtxt<'a, 'tcx, 'tcx>, def: DefId)-> ty::TraitDef<'tcx>;
    fn adt_def<'a>(&self, tcx: TyCtxt<'a, 'tcx, 'tcx>, def: DefId) -> ty::AdtDefMaster<'tcx>;
    fn fn_arg_names(&self, did: DefId) -> Vec<ast::Name>;
    fn inherent_implementations_for_type(&self, def_id: DefId) -> Vec<DefId>;

    // trait info
    fn implementations_of_trait(&self, filter: Option<DefId>) -> Vec<DefId>;

    // impl info
    fn impl_or_trait_items(&self, def_id: DefId) -> Vec<DefId>;
    fn impl_trait_ref<'a>(&self, tcx: TyCtxt<'a, 'tcx, 'tcx>, def: DefId)
                          -> Option<ty::TraitRef<'tcx>>;
    fn impl_polarity(&self, def: DefId) -> hir::ImplPolarity;
    fn custom_coerce_unsized_kind(&self, def: DefId)
                                  -> Option<ty::adjustment::CustomCoerceUnsized>;
    fn impl_parent(&self, impl_def_id: DefId) -> Option<DefId>;

    // trait/impl-item info
    fn trait_of_item(&self, def_id: DefId) -> Option<DefId>;
    fn impl_or_trait_item<'a>(&self, tcx: TyCtxt<'a, 'tcx, 'tcx>, def: DefId)
                              -> Option<ty::ImplOrTraitItem<'tcx>>;

    // flags
    fn is_const_fn(&self, did: DefId) -> bool;
    fn is_defaulted_trait(&self, did: DefId) -> bool;
    fn is_default_impl(&self, impl_did: DefId) -> bool;
    fn is_extern_item<'a>(&self, tcx: TyCtxt<'a, 'tcx, 'tcx>, did: DefId) -> bool;
    fn is_foreign_item(&self, did: DefId) -> bool;
    fn is_statically_included_foreign_item(&self, id: ast::NodeId) -> bool;

    // crate metadata
    fn dylib_dependency_formats(&self, cnum: CrateNum)
                                    -> Vec<(CrateNum, LinkagePreference)>;
    fn lang_items(&self, cnum: CrateNum) -> Vec<(DefIndex, usize)>;
    fn missing_lang_items(&self, cnum: CrateNum) -> Vec<lang_items::LangItem>;
    fn is_staged_api(&self, cnum: CrateNum) -> bool;
    fn is_explicitly_linked(&self, cnum: CrateNum) -> bool;
    fn is_allocator(&self, cnum: CrateNum) -> bool;
    fn is_panic_runtime(&self, cnum: CrateNum) -> bool;
    fn is_compiler_builtins(&self, cnum: CrateNum) -> bool;
    fn panic_strategy(&self, cnum: CrateNum) -> PanicStrategy;
    fn extern_crate(&self, cnum: CrateNum) -> Option<ExternCrate>;
    /// The name of the crate as it is referred to in source code of the current
    /// crate.
    fn crate_name(&self, cnum: CrateNum) -> InternedString;
    /// The name of the crate as it is stored in the crate's metadata.
    fn original_crate_name(&self, cnum: CrateNum) -> InternedString;
    fn crate_hash(&self, cnum: CrateNum) -> Svh;
    fn crate_disambiguator(&self, cnum: CrateNum) -> InternedString;
    fn plugin_registrar_fn(&self, cnum: CrateNum) -> Option<DefId>;
    fn native_libraries(&self, cnum: CrateNum) -> Vec<(NativeLibraryKind, String)>;
    fn reachable_ids(&self, cnum: CrateNum) -> Vec<DefId>;
    fn is_no_builtins(&self, cnum: CrateNum) -> bool;

    // resolve
    fn def_index_for_def_key(&self,
                             cnum: CrateNum,
                             def: DefKey)
                             -> Option<DefIndex>;
    fn def_key(&self, def: DefId) -> hir_map::DefKey;
    fn relative_def_path(&self, def: DefId) -> Option<hir_map::DefPath>;
    fn variant_kind(&self, def_id: DefId) -> Option<ty::VariantKind>;
    fn struct_ctor_def_id(&self, struct_def_id: DefId) -> Option<DefId>;
    fn struct_field_names(&self, def: DefId) -> Vec<ast::Name>;
    fn item_children(&self, did: DefId) -> Vec<def::Export>;

    // misc. metadata
    fn maybe_get_item_ast<'a>(&'tcx self, tcx: TyCtxt<'a, 'tcx, 'tcx>, def: DefId)
                              -> Option<(&'tcx InlinedItem, ast::NodeId)>;
    fn local_node_for_inlined_defid(&'tcx self, def_id: DefId) -> Option<ast::NodeId>;
    fn defid_for_inlined_node(&'tcx self, node_id: ast::NodeId) -> Option<DefId>;

    fn maybe_get_item_mir<'a>(&self, tcx: TyCtxt<'a, 'tcx, 'tcx>, def: DefId)
                              -> Option<Mir<'tcx>>;
    fn is_item_mir_available(&self, def: DefId) -> bool;

    // This is basically a 1-based range of ints, which is a little
    // silly - I may fix that.
    fn crates(&self) -> Vec<CrateNum>;
    fn used_libraries(&self) -> Vec<(String, NativeLibraryKind)>;
    fn used_link_args(&self) -> Vec<String>;

    // utility functions
    fn metadata_filename(&self) -> &str;
    fn metadata_section_name(&self, target: &Target) -> &str;
    fn used_crates(&self, prefer: LinkagePreference) -> Vec<(CrateNum, Option<PathBuf>)>;
    fn used_crate_source(&self, cnum: CrateNum) -> CrateSource;
    fn extern_mod_stmt_cnum(&self, emod_id: ast::NodeId) -> Option<CrateNum>;
    fn encode_metadata<'a>(&self, tcx: TyCtxt<'a, 'tcx, 'tcx>,
                           reexports: &def::ExportMap,
                           link_meta: &LinkMeta,
                           reachable: &NodeSet,
                           mir_map: &MirMap<'tcx>) -> Vec<u8>;
    fn metadata_encoding_version(&self) -> &[u8];
}

impl InlinedItem {
    pub fn visit<'ast,V>(&'ast self, visitor: &mut V)
        where V: Visitor<'ast>
    {
        match *self {
            InlinedItem::Item(_, ref i) => visitor.visit_item(&i),
            InlinedItem::TraitItem(_, ref ti) => visitor.visit_trait_item(ti),
            InlinedItem::ImplItem(_, ref ii) => visitor.visit_impl_item(ii),
        }
    }
}

// FIXME: find a better place for this?
pub fn validate_crate_name(sess: Option<&Session>, s: &str, sp: Option<Span>) {
    let mut err_count = 0;
    {
        let mut say = |s: &str| {
            match (sp, sess) {
                (_, None) => bug!("{}", s),
                (Some(sp), Some(sess)) => sess.span_err(sp, s),
                (None, Some(sess)) => sess.err(s),
            }
            err_count += 1;
        };
        if s.is_empty() {
            say("crate name must not be empty");
        }
        for c in s.chars() {
            if c.is_alphanumeric() { continue }
            if c == '_'  { continue }
            say(&format!("invalid character `{}` in crate name: `{}`", c, s));
        }
    }

    if err_count > 0 {
        sess.unwrap().abort_if_errors();
    }
}

/// A dummy crate store that does not support any non-local crates,
/// for test purposes.
pub struct DummyCrateStore;
#[allow(unused_variables)]
impl<'tcx> CrateStore<'tcx> for DummyCrateStore {
    // item info
    fn describe_def(&self, def: DefId) -> Option<Def> { bug!("describe_def") }
    fn stability(&self, def: DefId) -> Option<attr::Stability> { bug!("stability") }
    fn deprecation(&self, def: DefId) -> Option<attr::Deprecation> { bug!("deprecation") }
    fn visibility(&self, def: DefId) -> ty::Visibility { bug!("visibility") }
    fn closure_kind(&self, def_id: DefId) -> ty::ClosureKind { bug!("closure_kind") }
    fn closure_ty<'a>(&self, tcx: TyCtxt<'a, 'tcx, 'tcx>, def_id: DefId)
                      -> ty::ClosureTy<'tcx>  { bug!("closure_ty") }
    fn item_variances(&self, def: DefId) -> Vec<ty::Variance> { bug!("item_variances") }
    fn item_type<'a>(&self, tcx: TyCtxt<'a, 'tcx, 'tcx>, def: DefId)
                     -> Ty<'tcx> { bug!("item_type") }
    fn visible_parent_map<'a>(&'a self) -> ::std::cell::RefMut<'a, DefIdMap<DefId>> {
        bug!("visible_parent_map")
    }
    fn item_predicates<'a>(&self, tcx: TyCtxt<'a, 'tcx, 'tcx>, def: DefId)
                           -> ty::GenericPredicates<'tcx> { bug!("item_predicates") }
    fn item_super_predicates<'a>(&self, tcx: TyCtxt<'a, 'tcx, 'tcx>, def: DefId)
                                 -> ty::GenericPredicates<'tcx> { bug!("item_super_predicates") }
    fn item_generics<'a>(&self, tcx: TyCtxt<'a, 'tcx, 'tcx>, def: DefId)
                         -> ty::Generics<'tcx> { bug!("item_generics") }
    fn item_attrs(&self, def_id: DefId) -> Vec<ast::Attribute> { bug!("item_attrs") }
    fn trait_def<'a>(&self, tcx: TyCtxt<'a, 'tcx, 'tcx>, def: DefId)-> ty::TraitDef<'tcx>
        { bug!("trait_def") }
    fn adt_def<'a>(&self, tcx: TyCtxt<'a, 'tcx, 'tcx>, def: DefId) -> ty::AdtDefMaster<'tcx>
        { bug!("adt_def") }
    fn fn_arg_names(&self, did: DefId) -> Vec<ast::Name> { bug!("fn_arg_names") }
    fn inherent_implementations_for_type(&self, def_id: DefId) -> Vec<DefId> { vec![] }

    // trait info
    fn implementations_of_trait(&self, filter: Option<DefId>) -> Vec<DefId> { vec![] }
    fn def_index_for_def_key(&self,
                             cnum: CrateNum,
                             def: DefKey)
                             -> Option<DefIndex> {
        None
    }

    // impl info
    fn impl_or_trait_items(&self, def_id: DefId) -> Vec<DefId>
        { bug!("impl_or_trait_items") }
    fn impl_trait_ref<'a>(&self, tcx: TyCtxt<'a, 'tcx, 'tcx>, def: DefId)
                          -> Option<ty::TraitRef<'tcx>> { bug!("impl_trait_ref") }
    fn impl_polarity(&self, def: DefId) -> hir::ImplPolarity { bug!("impl_polarity") }
    fn custom_coerce_unsized_kind(&self, def: DefId)
                                  -> Option<ty::adjustment::CustomCoerceUnsized>
        { bug!("custom_coerce_unsized_kind") }
    fn impl_parent(&self, def: DefId) -> Option<DefId> { bug!("impl_parent") }

    // trait/impl-item info
    fn trait_of_item(&self, def_id: DefId) -> Option<DefId> { bug!("trait_of_item") }
    fn impl_or_trait_item<'a>(&self, tcx: TyCtxt<'a, 'tcx, 'tcx>, def: DefId)
                              -> Option<ty::ImplOrTraitItem<'tcx>> { bug!("impl_or_trait_item") }

    // flags
    fn is_const_fn(&self, did: DefId) -> bool { bug!("is_const_fn") }
    fn is_defaulted_trait(&self, did: DefId) -> bool { bug!("is_defaulted_trait") }
    fn is_default_impl(&self, impl_did: DefId) -> bool { bug!("is_default_impl") }
    fn is_extern_item<'a>(&self, tcx: TyCtxt<'a, 'tcx, 'tcx>, did: DefId) -> bool
        { bug!("is_extern_item") }
    fn is_foreign_item(&self, did: DefId) -> bool { bug!("is_foreign_item") }
    fn is_statically_included_foreign_item(&self, id: ast::NodeId) -> bool { false }

    // crate metadata
    fn dylib_dependency_formats(&self, cnum: CrateNum)
                                    -> Vec<(CrateNum, LinkagePreference)>
        { bug!("dylib_dependency_formats") }
    fn lang_items(&self, cnum: CrateNum) -> Vec<(DefIndex, usize)>
        { bug!("lang_items") }
    fn missing_lang_items(&self, cnum: CrateNum) -> Vec<lang_items::LangItem>
        { bug!("missing_lang_items") }
    fn is_staged_api(&self, cnum: CrateNum) -> bool { bug!("is_staged_api") }
    fn is_explicitly_linked(&self, cnum: CrateNum) -> bool { bug!("is_explicitly_linked") }
    fn is_allocator(&self, cnum: CrateNum) -> bool { bug!("is_allocator") }
    fn is_panic_runtime(&self, cnum: CrateNum) -> bool { bug!("is_panic_runtime") }
    fn is_compiler_builtins(&self, cnum: CrateNum) -> bool { bug!("is_compiler_builtins") }
    fn panic_strategy(&self, cnum: CrateNum) -> PanicStrategy {
        bug!("panic_strategy")
    }
    fn extern_crate(&self, cnum: CrateNum) -> Option<ExternCrate> { bug!("extern_crate") }
    fn crate_name(&self, cnum: CrateNum) -> InternedString { bug!("crate_name") }
    fn original_crate_name(&self, cnum: CrateNum) -> InternedString {
        bug!("original_crate_name")
    }
    fn crate_hash(&self, cnum: CrateNum) -> Svh { bug!("crate_hash") }
    fn crate_disambiguator(&self, cnum: CrateNum)
                           -> InternedString { bug!("crate_disambiguator") }
    fn plugin_registrar_fn(&self, cnum: CrateNum) -> Option<DefId>
        { bug!("plugin_registrar_fn") }
    fn native_libraries(&self, cnum: CrateNum) -> Vec<(NativeLibraryKind, String)>
        { bug!("native_libraries") }
    fn reachable_ids(&self, cnum: CrateNum) -> Vec<DefId> { bug!("reachable_ids") }
    fn is_no_builtins(&self, cnum: CrateNum) -> bool { bug!("is_no_builtins") }

    // resolve
    fn def_key(&self, def: DefId) -> hir_map::DefKey { bug!("def_key") }
    fn relative_def_path(&self, def: DefId) -> Option<hir_map::DefPath> {
        bug!("relative_def_path")
    }
    fn variant_kind(&self, def_id: DefId) -> Option<ty::VariantKind> { bug!("variant_kind") }
    fn struct_ctor_def_id(&self, struct_def_id: DefId) -> Option<DefId>
        { bug!("struct_ctor_def_id") }
    fn struct_field_names(&self, def: DefId) -> Vec<ast::Name> { bug!("struct_field_names") }
    fn item_children(&self, did: DefId) -> Vec<def::Export> { bug!("item_children") }

    // misc. metadata
    fn maybe_get_item_ast<'a>(&'tcx self, tcx: TyCtxt<'a, 'tcx, 'tcx>, def: DefId)
                              -> Option<(&'tcx InlinedItem, ast::NodeId)> {
        bug!("maybe_get_item_ast")
    }
    fn local_node_for_inlined_defid(&'tcx self, def_id: DefId) -> Option<ast::NodeId> {
        bug!("local_node_for_inlined_defid")
    }
    fn defid_for_inlined_node(&'tcx self, node_id: ast::NodeId) -> Option<DefId> {
        bug!("defid_for_inlined_node")
    }

    fn maybe_get_item_mir<'a>(&self, tcx: TyCtxt<'a, 'tcx, 'tcx>, def: DefId)
                              -> Option<Mir<'tcx>> { bug!("maybe_get_item_mir") }
    fn is_item_mir_available(&self, def: DefId) -> bool {
        bug!("is_item_mir_available")
    }

    // This is basically a 1-based range of ints, which is a little
    // silly - I may fix that.
    fn crates(&self) -> Vec<CrateNum> { vec![] }
    fn used_libraries(&self) -> Vec<(String, NativeLibraryKind)> { vec![] }
    fn used_link_args(&self) -> Vec<String> { vec![] }

    // utility functions
    fn metadata_filename(&self) -> &str { bug!("metadata_filename") }
    fn metadata_section_name(&self, target: &Target) -> &str { bug!("metadata_section_name") }
    fn used_crates(&self, prefer: LinkagePreference) -> Vec<(CrateNum, Option<PathBuf>)>
        { vec![] }
    fn used_crate_source(&self, cnum: CrateNum) -> CrateSource { bug!("used_crate_source") }
    fn extern_mod_stmt_cnum(&self, emod_id: ast::NodeId) -> Option<CrateNum> { None }
    fn encode_metadata<'a>(&self, tcx: TyCtxt<'a, 'tcx, 'tcx>,
                           reexports: &def::ExportMap,
                           link_meta: &LinkMeta,
                           reachable: &NodeSet,
                           mir_map: &MirMap<'tcx>) -> Vec<u8> { vec![] }
    fn metadata_encoding_version(&self) -> &[u8] { bug!("metadata_encoding_version") }
}

pub enum LoadedMacro {
    Def(ast::MacroDef),
    CustomDerive(String, Rc<MultiItemModifier>),
}

pub trait CrateLoader {
    fn load_macros(&mut self, extern_crate: &ast::Item, allows_macros: bool) -> Vec<LoadedMacro>;
    fn process_item(&mut self, item: &ast::Item, defs: &Definitions);
    fn postprocess(&mut self, krate: &ast::Crate);
}
