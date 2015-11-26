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

use back::svh::Svh;
use front::map as hir_map;
use middle::def;
use middle::lang_items;
use middle::ty::{self, Ty};
use middle::def_id::{DefId, DefIndex};
use session::Session;
use session::search_paths::PathKind;
use util::nodemap::{FnvHashMap, NodeMap, NodeSet};
use std::any::Any;
use std::cell::RefCell;
use std::rc::Rc;
use std::path::PathBuf;
use syntax::ast;
use syntax::ast_util::{IdVisitingOperation};
use syntax::attr;
use syntax::codemap::Span;
use syntax::ptr::P;
use rustc_back::target::Target;
use rustc_front::hir;
use rustc_front::intravisit::Visitor;
use rustc_front::util::IdVisitor;

pub use self::DefLike::{DlDef, DlField, DlImpl};
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
    pub cnum: ast::CrateNum,
}

#[derive(Copy, Debug, PartialEq, Clone)]
pub enum LinkagePreference {
    RequireDynamic,
    RequireStatic,
}

enum_from_u32! {
    #[derive(Copy, Clone, PartialEq)]
    pub enum NativeLibraryKind {
        NativeStatic,    // native static library (.a archive)
        NativeFramework, // OSX-specific
        NativeUnknown,   // default way to specify a dynamic library
    }
}

// Something that a name can resolve to.
#[derive(Copy, Clone, Debug)]
pub enum DefLike {
    DlDef(def::Def),
    DlImpl(DefId),
    DlField
}

/// The data we save and restore about an inlined item or method.  This is not
/// part of the AST that we parse from a file, but it becomes part of the tree
/// that we trans.
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub enum InlinedItem {
    Item(P<hir::Item>),
    TraitItem(DefId /* impl id */, P<hir::TraitItem>),
    ImplItem(DefId /* impl id */, P<hir::ImplItem>),
    Foreign(P<hir::ForeignItem>),
}

/// A borrowed version of `hir::InlinedItem`.
pub enum InlinedItemRef<'a> {
    Item(&'a hir::Item),
    TraitItem(DefId, &'a hir::TraitItem),
    ImplItem(DefId, &'a hir::ImplItem),
    Foreign(&'a hir::ForeignItem)
}

/// Item definitions in the currently-compiled crate would have the CrateNum
/// LOCAL_CRATE in their DefId.
pub const LOCAL_CRATE: ast::CrateNum = 0;

pub struct ChildItem {
    pub def: DefLike,
    pub name: ast::Name,
    pub vis: hir::Visibility
}

pub enum FoundAst<'ast> {
    Found(&'ast InlinedItem),
    FoundParent(DefId, &'ast InlinedItem),
    NotFound,
}

/// A store of Rust crates, through with their metadata
/// can be accessed.
///
/// The `: Any` bound is a temporary measure that allows access
/// to the backing `rustc_metadata::cstore::CStore` object. It
/// will be removed in the near future - if you need to access
/// internal APIs, please tell us.
pub trait CrateStore<'tcx> : Any {
    // item info
    fn stability(&self, def: DefId) -> Option<attr::Stability>;
    fn closure_kind(&self, tcx: &ty::ctxt<'tcx>, def_id: DefId)
                    -> ty::ClosureKind;
    fn closure_ty(&self, tcx: &ty::ctxt<'tcx>, def_id: DefId)
                  -> ty::ClosureTy<'tcx>;
    fn item_variances(&self, def: DefId) -> ty::ItemVariances;
    fn repr_attrs(&self, def: DefId) -> Vec<attr::ReprAttr>;
    fn item_type(&self, tcx: &ty::ctxt<'tcx>, def: DefId)
                 -> ty::TypeScheme<'tcx>;
    fn item_path(&self, def: DefId) -> Vec<hir_map::PathElem>;
    fn item_name(&self, def: DefId) -> ast::Name;
    fn item_predicates(&self, tcx: &ty::ctxt<'tcx>, def: DefId)
                       -> ty::GenericPredicates<'tcx>;
    fn item_super_predicates(&self, tcx: &ty::ctxt<'tcx>, def: DefId)
                             -> ty::GenericPredicates<'tcx>;
    fn item_attrs(&self, def_id: DefId) -> Vec<ast::Attribute>;
    fn item_symbol(&self, def: DefId) -> String;
    fn trait_def(&self, tcx: &ty::ctxt<'tcx>, def: DefId)-> ty::TraitDef<'tcx>;
    fn adt_def(&self, tcx: &ty::ctxt<'tcx>, def: DefId) -> ty::AdtDefMaster<'tcx>;
    fn method_arg_names(&self, did: DefId) -> Vec<String>;
    fn inherent_implementations_for_type(&self, def_id: DefId) -> Vec<DefId>;

    // trait info
    fn implementations_of_trait(&self, def_id: DefId) -> Vec<DefId>;
    fn provided_trait_methods(&self, tcx: &ty::ctxt<'tcx>, def: DefId)
                              -> Vec<Rc<ty::Method<'tcx>>>;
    fn trait_item_def_ids(&self, def: DefId)
                          -> Vec<ty::ImplOrTraitItemId>;

    // impl info
    fn impl_items(&self, impl_def_id: DefId) -> Vec<ty::ImplOrTraitItemId>;
    fn impl_trait_ref(&self, tcx: &ty::ctxt<'tcx>, def: DefId)
                      -> Option<ty::TraitRef<'tcx>>;
    fn impl_polarity(&self, def: DefId) -> Option<hir::ImplPolarity>;
    fn custom_coerce_unsized_kind(&self, def: DefId)
                                  -> Option<ty::adjustment::CustomCoerceUnsized>;
    fn associated_consts(&self, tcx: &ty::ctxt<'tcx>, def: DefId)
                         -> Vec<Rc<ty::AssociatedConst<'tcx>>>;

    // trait/impl-item info
    fn trait_of_item(&self, tcx: &ty::ctxt<'tcx>, def_id: DefId)
                     -> Option<DefId>;
    fn impl_or_trait_item(&self, tcx: &ty::ctxt<'tcx>, def: DefId)
                          -> ty::ImplOrTraitItem<'tcx>;

    // flags
    fn is_const_fn(&self, did: DefId) -> bool;
    fn is_defaulted_trait(&self, did: DefId) -> bool;
    fn is_impl(&self, did: DefId) -> bool;
    fn is_default_impl(&self, impl_did: DefId) -> bool;
    fn is_extern_fn(&self, tcx: &ty::ctxt<'tcx>, did: DefId) -> bool;
    fn is_static(&self, did: DefId) -> bool;
    fn is_static_method(&self, did: DefId) -> bool;
    fn is_statically_included_foreign_item(&self, id: ast::NodeId) -> bool;
    fn is_typedef(&self, did: DefId) -> bool;

    // crate metadata
    fn dylib_dependency_formats(&self, cnum: ast::CrateNum)
                                    -> Vec<(ast::CrateNum, LinkagePreference)>;
    fn lang_items(&self, cnum: ast::CrateNum) -> Vec<(DefIndex, usize)>;
    fn missing_lang_items(&self, cnum: ast::CrateNum) -> Vec<lang_items::LangItem>;
    fn is_staged_api(&self, cnum: ast::CrateNum) -> bool;
    fn is_explicitly_linked(&self, cnum: ast::CrateNum) -> bool;
    fn is_allocator(&self, cnum: ast::CrateNum) -> bool;
    fn crate_attrs(&self, cnum: ast::CrateNum) -> Vec<ast::Attribute>;
    fn crate_name(&self, cnum: ast::CrateNum) -> String;
    fn crate_hash(&self, cnum: ast::CrateNum) -> Svh;
    fn crate_struct_field_attrs(&self, cnum: ast::CrateNum)
                                -> FnvHashMap<DefId, Vec<ast::Attribute>>;
    fn plugin_registrar_fn(&self, cnum: ast::CrateNum) -> Option<DefId>;
    fn native_libraries(&self, cnum: ast::CrateNum) -> Vec<(NativeLibraryKind, String)>;
    fn reachable_ids(&self, cnum: ast::CrateNum) -> Vec<DefId>;

    // resolve
    fn def_path(&self, def: DefId) -> hir_map::DefPath;
    fn tuple_struct_definition_if_ctor(&self, did: DefId) -> Option<DefId>;
    fn struct_field_names(&self, def: DefId) -> Vec<ast::Name>;
    fn item_children(&self, did: DefId) -> Vec<ChildItem>;
    fn crate_top_level_items(&self, cnum: ast::CrateNum) -> Vec<ChildItem>;

    // misc. metadata
    fn maybe_get_item_ast(&'tcx self, tcx: &ty::ctxt<'tcx>, def: DefId)
                          -> FoundAst<'tcx>;
    // This is basically a 1-based range of ints, which is a little
    // silly - I may fix that.
    fn crates(&self) -> Vec<ast::CrateNum>;
    fn used_libraries(&self) -> Vec<(String, NativeLibraryKind)>;
    fn used_link_args(&self) -> Vec<String>;

    // utility functions
    fn metadata_filename(&self) -> &str;
    fn metadata_section_name(&self, target: &Target) -> &str;
    fn encode_type(&self, tcx: &ty::ctxt<'tcx>, ty: Ty<'tcx>) -> Vec<u8>;
    fn used_crates(&self, prefer: LinkagePreference) -> Vec<(ast::CrateNum, Option<PathBuf>)>;
    fn used_crate_source(&self, cnum: ast::CrateNum) -> CrateSource;
    fn extern_mod_stmt_cnum(&self, emod_id: ast::NodeId) -> Option<ast::CrateNum>;
    fn encode_metadata(&self,
                       tcx: &ty::ctxt<'tcx>,
                       reexports: &def::ExportMap,
                       item_symbols: &RefCell<NodeMap<String>>,
                       link_meta: &LinkMeta,
                       reachable: &NodeSet,
                       krate: &hir::Crate) -> Vec<u8>;
    fn metadata_encoding_version(&self) -> &[u8];
}

impl InlinedItem {
    pub fn visit<'ast,V>(&'ast self, visitor: &mut V)
        where V: Visitor<'ast>
    {
        match *self {
            InlinedItem::Item(ref i) => visitor.visit_item(&**i),
            InlinedItem::Foreign(ref i) => visitor.visit_foreign_item(&**i),
            InlinedItem::TraitItem(_, ref ti) => visitor.visit_trait_item(ti),
            InlinedItem::ImplItem(_, ref ii) => visitor.visit_impl_item(ii),
        }
    }

    pub fn visit_ids<O: IdVisitingOperation>(&self, operation: &mut O) {
        let mut id_visitor = IdVisitor::new(operation);
        self.visit(&mut id_visitor);
    }
}

// FIXME: find a better place for this?
pub fn validate_crate_name(sess: Option<&Session>, s: &str, sp: Option<Span>) {
    let say = |s: &str| {
        match (sp, sess) {
            (_, None) => panic!("{}", s),
            (Some(sp), Some(sess)) => sess.span_err(sp, s),
            (None, Some(sess)) => sess.err(s),
        }
    };
    if s.is_empty() {
        say("crate name must not be empty");
    }
    for c in s.chars() {
        if c.is_alphanumeric() { continue }
        if c == '_'  { continue }
        say(&format!("invalid character `{}` in crate name: `{}`", c, s));
    }
    match sess {
        Some(sess) => sess.abort_if_errors(),
        None => {}
    }
}

/// A dummy crate store that does not support any non-local crates,
/// for test purposes.
pub struct DummyCrateStore;
#[allow(unused_variables)]
impl<'tcx> CrateStore<'tcx> for DummyCrateStore {
    // item info
    fn stability(&self, def: DefId) -> Option<attr::Stability> { unimplemented!() }
    fn closure_kind(&self, tcx: &ty::ctxt<'tcx>, def_id: DefId)
                    -> ty::ClosureKind  { unimplemented!() }
    fn closure_ty(&self, tcx: &ty::ctxt<'tcx>, def_id: DefId)
                  -> ty::ClosureTy<'tcx>  { unimplemented!() }
    fn item_variances(&self, def: DefId) -> ty::ItemVariances { unimplemented!() }
    fn repr_attrs(&self, def: DefId) -> Vec<attr::ReprAttr> { unimplemented!() }
    fn item_type(&self, tcx: &ty::ctxt<'tcx>, def: DefId)
                 -> ty::TypeScheme<'tcx> { unimplemented!() }
    fn item_path(&self, def: DefId) -> Vec<hir_map::PathElem> { unimplemented!() }
    fn item_name(&self, def: DefId) -> ast::Name { unimplemented!() }
    fn item_predicates(&self, tcx: &ty::ctxt<'tcx>, def: DefId)
                       -> ty::GenericPredicates<'tcx> { unimplemented!() }
    fn item_super_predicates(&self, tcx: &ty::ctxt<'tcx>, def: DefId)
                             -> ty::GenericPredicates<'tcx> { unimplemented!() }
    fn item_attrs(&self, def_id: DefId) -> Vec<ast::Attribute> { unimplemented!() }
    fn item_symbol(&self, def: DefId) -> String { unimplemented!() }
    fn trait_def(&self, tcx: &ty::ctxt<'tcx>, def: DefId)-> ty::TraitDef<'tcx>
        { unimplemented!() }
    fn adt_def(&self, tcx: &ty::ctxt<'tcx>, def: DefId) -> ty::AdtDefMaster<'tcx>
        { unimplemented!() }
    fn method_arg_names(&self, did: DefId) -> Vec<String> { unimplemented!() }
    fn inherent_implementations_for_type(&self, def_id: DefId) -> Vec<DefId> { vec![] }

    // trait info
    fn implementations_of_trait(&self, def_id: DefId) -> Vec<DefId> { vec![] }
    fn provided_trait_methods(&self, tcx: &ty::ctxt<'tcx>, def: DefId)
                              -> Vec<Rc<ty::Method<'tcx>>> { unimplemented!() }
    fn trait_item_def_ids(&self, def: DefId)
                          -> Vec<ty::ImplOrTraitItemId> { unimplemented!() }

    // impl info
    fn impl_items(&self, impl_def_id: DefId) -> Vec<ty::ImplOrTraitItemId>
        { unimplemented!() }
    fn impl_trait_ref(&self, tcx: &ty::ctxt<'tcx>, def: DefId)
                      -> Option<ty::TraitRef<'tcx>> { unimplemented!() }
    fn impl_polarity(&self, def: DefId) -> Option<hir::ImplPolarity> { unimplemented!() }
    fn custom_coerce_unsized_kind(&self, def: DefId)
                                  -> Option<ty::adjustment::CustomCoerceUnsized>
        { unimplemented!() }
    fn associated_consts(&self, tcx: &ty::ctxt<'tcx>, def: DefId)
                         -> Vec<Rc<ty::AssociatedConst<'tcx>>> { unimplemented!() }

    // trait/impl-item info
    fn trait_of_item(&self, tcx: &ty::ctxt<'tcx>, def_id: DefId)
                     -> Option<DefId> { unimplemented!() }
    fn impl_or_trait_item(&self, tcx: &ty::ctxt<'tcx>, def: DefId)
                          -> ty::ImplOrTraitItem<'tcx> { unimplemented!() }

    // flags
    fn is_const_fn(&self, did: DefId) -> bool { unimplemented!() }
    fn is_defaulted_trait(&self, did: DefId) -> bool { unimplemented!() }
    fn is_impl(&self, did: DefId) -> bool { unimplemented!() }
    fn is_default_impl(&self, impl_did: DefId) -> bool { unimplemented!() }
    fn is_extern_fn(&self, tcx: &ty::ctxt<'tcx>, did: DefId) -> bool { unimplemented!() }
    fn is_static(&self, did: DefId) -> bool { unimplemented!() }
    fn is_static_method(&self, did: DefId) -> bool { unimplemented!() }
    fn is_statically_included_foreign_item(&self, id: ast::NodeId) -> bool { false }
    fn is_typedef(&self, did: DefId) -> bool { unimplemented!() }

    // crate metadata
    fn dylib_dependency_formats(&self, cnum: ast::CrateNum)
                                    -> Vec<(ast::CrateNum, LinkagePreference)>
        { unimplemented!() }
    fn lang_items(&self, cnum: ast::CrateNum) -> Vec<(DefIndex, usize)>
        { unimplemented!() }
    fn missing_lang_items(&self, cnum: ast::CrateNum) -> Vec<lang_items::LangItem>
        { unimplemented!() }
    fn is_staged_api(&self, cnum: ast::CrateNum) -> bool { unimplemented!() }
    fn is_explicitly_linked(&self, cnum: ast::CrateNum) -> bool { unimplemented!() }
    fn is_allocator(&self, cnum: ast::CrateNum) -> bool { unimplemented!() }
    fn crate_attrs(&self, cnum: ast::CrateNum) -> Vec<ast::Attribute>
        { unimplemented!() }
    fn crate_name(&self, cnum: ast::CrateNum) -> String { unimplemented!() }
    fn crate_hash(&self, cnum: ast::CrateNum) -> Svh { unimplemented!() }
    fn crate_struct_field_attrs(&self, cnum: ast::CrateNum)
                                -> FnvHashMap<DefId, Vec<ast::Attribute>>
        { unimplemented!() }
    fn plugin_registrar_fn(&self, cnum: ast::CrateNum) -> Option<DefId>
        { unimplemented!() }
    fn native_libraries(&self, cnum: ast::CrateNum) -> Vec<(NativeLibraryKind, String)>
        { unimplemented!() }
    fn reachable_ids(&self, cnum: ast::CrateNum) -> Vec<DefId> { unimplemented!() }

    // resolve
    fn def_path(&self, def: DefId) -> hir_map::DefPath { unimplemented!() }
    fn tuple_struct_definition_if_ctor(&self, did: DefId) -> Option<DefId>
        { unimplemented!() }
    fn struct_field_names(&self, def: DefId) -> Vec<ast::Name> { unimplemented!() }
    fn item_children(&self, did: DefId) -> Vec<ChildItem> { unimplemented!() }
    fn crate_top_level_items(&self, cnum: ast::CrateNum) -> Vec<ChildItem>
        { unimplemented!() }

    // misc. metadata
    fn maybe_get_item_ast(&'tcx self, tcx: &ty::ctxt<'tcx>, def: DefId)
                          -> FoundAst<'tcx> { unimplemented!() }
    // This is basically a 1-based range of ints, which is a little
    // silly - I may fix that.
    fn crates(&self) -> Vec<ast::CrateNum> { vec![] }
    fn used_libraries(&self) -> Vec<(String, NativeLibraryKind)> { vec![] }
    fn used_link_args(&self) -> Vec<String> { vec![] }

    // utility functions
    fn metadata_filename(&self) -> &str { unimplemented!() }
    fn metadata_section_name(&self, target: &Target) -> &str { unimplemented!() }
    fn encode_type(&self, tcx: &ty::ctxt<'tcx>, ty: Ty<'tcx>) -> Vec<u8>
        { unimplemented!() }
    fn used_crates(&self, prefer: LinkagePreference) -> Vec<(ast::CrateNum, Option<PathBuf>)>
        { vec![] }
    fn used_crate_source(&self, cnum: ast::CrateNum) -> CrateSource { unimplemented!() }
    fn extern_mod_stmt_cnum(&self, emod_id: ast::NodeId) -> Option<ast::CrateNum> { None }
    fn encode_metadata(&self,
                       tcx: &ty::ctxt<'tcx>,
                       reexports: &def::ExportMap,
                       item_symbols: &RefCell<NodeMap<String>>,
                       link_meta: &LinkMeta,
                       reachable: &NodeSet,
                       krate: &hir::Crate) -> Vec<u8> { vec![] }
    fn metadata_encoding_version(&self) -> &[u8] { unimplemented!() }
}
