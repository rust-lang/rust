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
use middle::def::{self, Def};
use middle::lang_items;
use middle::ty::{self, Ty, TyCtxt, VariantKind};
use middle::def_id::{DefId, DefIndex};
use mir::repr::Mir;
use mir::mir_map::MirMap;
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
use syntax::parse::token::InternedString;
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
    DlDef(Def),
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
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
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
    FoundParent(DefId, &'ast hir::Item),
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
    fn deprecation(&self, def: DefId) -> Option<attr::Deprecation>;
    fn visibility(&self, def: DefId) -> hir::Visibility;
    fn closure_kind(&self, tcx: &TyCtxt<'tcx>, def_id: DefId)
                    -> ty::ClosureKind;
    fn closure_ty(&self, tcx: &TyCtxt<'tcx>, def_id: DefId)
                  -> ty::ClosureTy<'tcx>;
    fn item_variances(&self, def: DefId) -> ty::ItemVariances;
    fn repr_attrs(&self, def: DefId) -> Vec<attr::ReprAttr>;
    fn item_type(&self, tcx: &TyCtxt<'tcx>, def: DefId)
                 -> ty::TypeScheme<'tcx>;
    fn item_path(&self, def: DefId) -> Vec<hir_map::PathElem>;
    fn extern_item_path(&self, def: DefId) -> Vec<hir_map::PathElem>;
    fn item_name(&self, def: DefId) -> ast::Name;
    fn item_predicates(&self, tcx: &TyCtxt<'tcx>, def: DefId)
                       -> ty::GenericPredicates<'tcx>;
    fn item_super_predicates(&self, tcx: &TyCtxt<'tcx>, def: DefId)
                             -> ty::GenericPredicates<'tcx>;
    fn item_attrs(&self, def_id: DefId) -> Vec<ast::Attribute>;
    fn item_symbol(&self, def: DefId) -> String;
    fn trait_def(&self, tcx: &TyCtxt<'tcx>, def: DefId)-> ty::TraitDef<'tcx>;
    fn adt_def(&self, tcx: &TyCtxt<'tcx>, def: DefId) -> ty::AdtDefMaster<'tcx>;
    fn method_arg_names(&self, did: DefId) -> Vec<String>;
    fn inherent_implementations_for_type(&self, def_id: DefId) -> Vec<DefId>;

    // trait info
    fn implementations_of_trait(&self, def_id: DefId) -> Vec<DefId>;
    fn provided_trait_methods(&self, tcx: &TyCtxt<'tcx>, def: DefId)
                              -> Vec<Rc<ty::Method<'tcx>>>;
    fn trait_item_def_ids(&self, def: DefId)
                          -> Vec<ty::ImplOrTraitItemId>;

    // impl info
    fn impl_items(&self, impl_def_id: DefId) -> Vec<ty::ImplOrTraitItemId>;
    fn impl_trait_ref(&self, tcx: &TyCtxt<'tcx>, def: DefId)
                      -> Option<ty::TraitRef<'tcx>>;
    fn impl_polarity(&self, def: DefId) -> Option<hir::ImplPolarity>;
    fn custom_coerce_unsized_kind(&self, def: DefId)
                                  -> Option<ty::adjustment::CustomCoerceUnsized>;
    fn associated_consts(&self, tcx: &TyCtxt<'tcx>, def: DefId)
                         -> Vec<Rc<ty::AssociatedConst<'tcx>>>;
    fn impl_parent(&self, impl_def_id: DefId) -> Option<DefId>;

    // trait/impl-item info
    fn trait_of_item(&self, tcx: &TyCtxt<'tcx>, def_id: DefId)
                     -> Option<DefId>;
    fn impl_or_trait_item(&self, tcx: &TyCtxt<'tcx>, def: DefId)
                          -> Option<ty::ImplOrTraitItem<'tcx>>;

    // flags
    fn is_const_fn(&self, did: DefId) -> bool;
    fn is_defaulted_trait(&self, did: DefId) -> bool;
    fn is_impl(&self, did: DefId) -> bool;
    fn is_default_impl(&self, impl_did: DefId) -> bool;
    fn is_extern_item(&self, tcx: &TyCtxt<'tcx>, did: DefId) -> bool;
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
    fn crate_name(&self, cnum: ast::CrateNum) -> InternedString;
    fn crate_hash(&self, cnum: ast::CrateNum) -> Svh;
    fn crate_disambiguator(&self, cnum: ast::CrateNum) -> InternedString;
    fn crate_struct_field_attrs(&self, cnum: ast::CrateNum)
                                -> FnvHashMap<DefId, Vec<ast::Attribute>>;
    fn plugin_registrar_fn(&self, cnum: ast::CrateNum) -> Option<DefId>;
    fn native_libraries(&self, cnum: ast::CrateNum) -> Vec<(NativeLibraryKind, String)>;
    fn reachable_ids(&self, cnum: ast::CrateNum) -> Vec<DefId>;

    // resolve
    fn def_path(&self, def: DefId) -> hir_map::DefPath;
    fn variant_kind(&self, def_id: DefId) -> Option<VariantKind>;
    fn struct_ctor_def_id(&self, struct_def_id: DefId) -> Option<DefId>;
    fn tuple_struct_definition_if_ctor(&self, did: DefId) -> Option<DefId>;
    fn struct_field_names(&self, def: DefId) -> Vec<ast::Name>;
    fn item_children(&self, did: DefId) -> Vec<ChildItem>;
    fn crate_top_level_items(&self, cnum: ast::CrateNum) -> Vec<ChildItem>;

    // misc. metadata
    fn maybe_get_item_ast(&'tcx self, tcx: &TyCtxt<'tcx>, def: DefId)
                          -> FoundAst<'tcx>;
    fn maybe_get_item_mir(&self, tcx: &TyCtxt<'tcx>, def: DefId)
                          -> Option<Mir<'tcx>>;
    fn is_item_mir_available(&self, def: DefId) -> bool;

    // This is basically a 1-based range of ints, which is a little
    // silly - I may fix that.
    fn crates(&self) -> Vec<ast::CrateNum>;
    fn used_libraries(&self) -> Vec<(String, NativeLibraryKind)>;
    fn used_link_args(&self) -> Vec<String>;

    // utility functions
    fn metadata_filename(&self) -> &str;
    fn metadata_section_name(&self, target: &Target) -> &str;
    fn encode_type(&self,
                   tcx: &TyCtxt<'tcx>,
                   ty: Ty<'tcx>,
                   def_id_to_string: fn(&TyCtxt<'tcx>, DefId) -> String)
                   -> Vec<u8>;
    fn used_crates(&self, prefer: LinkagePreference) -> Vec<(ast::CrateNum, Option<PathBuf>)>;
    fn used_crate_source(&self, cnum: ast::CrateNum) -> CrateSource;
    fn extern_mod_stmt_cnum(&self, emod_id: ast::NodeId) -> Option<ast::CrateNum>;
    fn encode_metadata(&self,
                       tcx: &TyCtxt<'tcx>,
                       reexports: &def::ExportMap,
                       item_symbols: &RefCell<NodeMap<String>>,
                       link_meta: &LinkMeta,
                       reachable: &NodeSet,
                       mir_map: &MirMap<'tcx>,
                       krate: &hir::Crate) -> Vec<u8>;
    fn metadata_encoding_version(&self) -> &[u8];
}

impl InlinedItem {
    pub fn visit<'ast,V>(&'ast self, visitor: &mut V)
        where V: Visitor<'ast>
    {
        match *self {
            InlinedItem::Item(ref i) => visitor.visit_item(&i),
            InlinedItem::Foreign(ref i) => visitor.visit_foreign_item(&i),
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
    let mut err_count = 0;
    {
        let mut say = |s: &str| {
            match (sp, sess) {
                (_, None) => panic!("{}", s),
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
    fn stability(&self, def: DefId) -> Option<attr::Stability> { unimplemented!() }
    fn deprecation(&self, def: DefId) -> Option<attr::Deprecation> { unimplemented!() }
    fn visibility(&self, def: DefId) -> hir::Visibility { unimplemented!() }
    fn closure_kind(&self, tcx: &TyCtxt<'tcx>, def_id: DefId)
                    -> ty::ClosureKind  { unimplemented!() }
    fn closure_ty(&self, tcx: &TyCtxt<'tcx>, def_id: DefId)
                  -> ty::ClosureTy<'tcx>  { unimplemented!() }
    fn item_variances(&self, def: DefId) -> ty::ItemVariances { unimplemented!() }
    fn repr_attrs(&self, def: DefId) -> Vec<attr::ReprAttr> { unimplemented!() }
    fn item_type(&self, tcx: &TyCtxt<'tcx>, def: DefId)
                 -> ty::TypeScheme<'tcx> { unimplemented!() }
    fn item_path(&self, def: DefId) -> Vec<hir_map::PathElem> { unimplemented!() }
    fn extern_item_path(&self, def: DefId) -> Vec<hir_map::PathElem> { unimplemented!() }
    fn item_name(&self, def: DefId) -> ast::Name { unimplemented!() }
    fn item_predicates(&self, tcx: &TyCtxt<'tcx>, def: DefId)
                       -> ty::GenericPredicates<'tcx> { unimplemented!() }
    fn item_super_predicates(&self, tcx: &TyCtxt<'tcx>, def: DefId)
                             -> ty::GenericPredicates<'tcx> { unimplemented!() }
    fn item_attrs(&self, def_id: DefId) -> Vec<ast::Attribute> { unimplemented!() }
    fn item_symbol(&self, def: DefId) -> String { unimplemented!() }
    fn trait_def(&self, tcx: &TyCtxt<'tcx>, def: DefId)-> ty::TraitDef<'tcx>
        { unimplemented!() }
    fn adt_def(&self, tcx: &TyCtxt<'tcx>, def: DefId) -> ty::AdtDefMaster<'tcx>
        { unimplemented!() }
    fn method_arg_names(&self, did: DefId) -> Vec<String> { unimplemented!() }
    fn inherent_implementations_for_type(&self, def_id: DefId) -> Vec<DefId> { vec![] }

    // trait info
    fn implementations_of_trait(&self, def_id: DefId) -> Vec<DefId> { vec![] }
    fn provided_trait_methods(&self, tcx: &TyCtxt<'tcx>, def: DefId)
                              -> Vec<Rc<ty::Method<'tcx>>> { unimplemented!() }
    fn trait_item_def_ids(&self, def: DefId)
                          -> Vec<ty::ImplOrTraitItemId> { unimplemented!() }

    // impl info
    fn impl_items(&self, impl_def_id: DefId) -> Vec<ty::ImplOrTraitItemId>
        { unimplemented!() }
    fn impl_trait_ref(&self, tcx: &TyCtxt<'tcx>, def: DefId)
                      -> Option<ty::TraitRef<'tcx>> { unimplemented!() }
    fn impl_polarity(&self, def: DefId) -> Option<hir::ImplPolarity> { unimplemented!() }
    fn custom_coerce_unsized_kind(&self, def: DefId)
                                  -> Option<ty::adjustment::CustomCoerceUnsized>
        { unimplemented!() }
    fn associated_consts(&self, tcx: &TyCtxt<'tcx>, def: DefId)
                         -> Vec<Rc<ty::AssociatedConst<'tcx>>> { unimplemented!() }
    fn impl_parent(&self, def: DefId) -> Option<DefId> { unimplemented!() }

    // trait/impl-item info
    fn trait_of_item(&self, tcx: &TyCtxt<'tcx>, def_id: DefId)
                     -> Option<DefId> { unimplemented!() }
    fn impl_or_trait_item(&self, tcx: &TyCtxt<'tcx>, def: DefId)
                          -> Option<ty::ImplOrTraitItem<'tcx>> { unimplemented!() }

    // flags
    fn is_const_fn(&self, did: DefId) -> bool { unimplemented!() }
    fn is_defaulted_trait(&self, did: DefId) -> bool { unimplemented!() }
    fn is_impl(&self, did: DefId) -> bool { unimplemented!() }
    fn is_default_impl(&self, impl_did: DefId) -> bool { unimplemented!() }
    fn is_extern_item(&self, tcx: &TyCtxt<'tcx>, did: DefId) -> bool { unimplemented!() }
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
    fn crate_name(&self, cnum: ast::CrateNum) -> InternedString { unimplemented!() }
    fn crate_hash(&self, cnum: ast::CrateNum) -> Svh { unimplemented!() }
    fn crate_disambiguator(&self, cnum: ast::CrateNum) -> InternedString { unimplemented!() }
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
    fn variant_kind(&self, def_id: DefId) -> Option<VariantKind> { unimplemented!() }
    fn struct_ctor_def_id(&self, struct_def_id: DefId) -> Option<DefId>
        { unimplemented!() }
    fn tuple_struct_definition_if_ctor(&self, did: DefId) -> Option<DefId>
        { unimplemented!() }
    fn struct_field_names(&self, def: DefId) -> Vec<ast::Name> { unimplemented!() }
    fn item_children(&self, did: DefId) -> Vec<ChildItem> { unimplemented!() }
    fn crate_top_level_items(&self, cnum: ast::CrateNum) -> Vec<ChildItem>
        { unimplemented!() }

    // misc. metadata
    fn maybe_get_item_ast(&'tcx self, tcx: &TyCtxt<'tcx>, def: DefId)
                          -> FoundAst<'tcx> { unimplemented!() }
    fn maybe_get_item_mir(&self, tcx: &TyCtxt<'tcx>, def: DefId)
                          -> Option<Mir<'tcx>> { unimplemented!() }
    fn is_item_mir_available(&self, def: DefId) -> bool {
        unimplemented!()
    }

    // This is basically a 1-based range of ints, which is a little
    // silly - I may fix that.
    fn crates(&self) -> Vec<ast::CrateNum> { vec![] }
    fn used_libraries(&self) -> Vec<(String, NativeLibraryKind)> { vec![] }
    fn used_link_args(&self) -> Vec<String> { vec![] }

    // utility functions
    fn metadata_filename(&self) -> &str { unimplemented!() }
    fn metadata_section_name(&self, target: &Target) -> &str { unimplemented!() }
    fn encode_type(&self,
                   tcx: &TyCtxt<'tcx>,
                   ty: Ty<'tcx>,
                   def_id_to_string: fn(&TyCtxt<'tcx>, DefId) -> String)
                   -> Vec<u8> {
        unimplemented!()
    }
    fn used_crates(&self, prefer: LinkagePreference) -> Vec<(ast::CrateNum, Option<PathBuf>)>
        { vec![] }
    fn used_crate_source(&self, cnum: ast::CrateNum) -> CrateSource { unimplemented!() }
    fn extern_mod_stmt_cnum(&self, emod_id: ast::NodeId) -> Option<ast::CrateNum> { None }
    fn encode_metadata(&self,
                       tcx: &TyCtxt<'tcx>,
                       reexports: &def::ExportMap,
                       item_symbols: &RefCell<NodeMap<String>>,
                       link_meta: &LinkMeta,
                       reachable: &NodeSet,
                       mir_map: &MirMap<'tcx>,
                       krate: &hir::Crate) -> Vec<u8> { vec![] }
    fn metadata_encoding_version(&self) -> &[u8] { unimplemented!() }
}


/// Metadata encoding and decoding can make use of thread-local encoding and
/// decoding contexts. These allow implementers of serialize::Encodable and
/// Decodable to access information and datastructures that would otherwise not
/// be available to them. For example, we can automatically translate def-id and
/// span information during decoding because the decoding context knows which
/// crate the data is decoded from. Or it allows to make ty::Ty decodable
/// because the context has access to the TyCtxt that is needed for creating
/// ty::Ty instances.
///
/// Note, however, that this only works for RBML-based encoding and decoding at
/// the moment.
pub mod tls {
    use rbml::opaque::Encoder as OpaqueEncoder;
    use rbml::opaque::Decoder as OpaqueDecoder;
    use serialize;
    use std::cell::Cell;
    use std::mem;
    use middle::ty::{self, Ty, TyCtxt};
    use middle::subst::Substs;
    use middle::def_id::DefId;

    pub trait EncodingContext<'tcx> {
        fn tcx<'a>(&'a self) -> &'a TyCtxt<'tcx>;
        fn encode_ty(&self, encoder: &mut OpaqueEncoder, t: Ty<'tcx>);
        fn encode_substs(&self, encoder: &mut OpaqueEncoder, substs: &Substs<'tcx>);
    }

    /// Marker type used for the TLS slot.
    /// The type context cannot be used directly because the TLS
    /// in libstd doesn't allow types generic over lifetimes.
    struct TlsPayload;

    thread_local! {
        static TLS_ENCODING: Cell<Option<*const TlsPayload>> = Cell::new(None)
    }

    /// Execute f after pushing the given EncodingContext onto the TLS stack.
    pub fn enter_encoding_context<'tcx, F, R>(ecx: &EncodingContext<'tcx>,
                                              encoder: &mut OpaqueEncoder,
                                              f: F) -> R
        where F: FnOnce(&EncodingContext<'tcx>, &mut OpaqueEncoder) -> R
    {
        let tls_payload = (ecx as *const _, encoder as *mut _);
        let tls_ptr = &tls_payload as *const _ as *const TlsPayload;
        TLS_ENCODING.with(|tls| {
            let prev = tls.get();
            tls.set(Some(tls_ptr));
            let ret = f(ecx, encoder);
            tls.set(prev);
            return ret
        })
    }

    /// Execute f with access to the thread-local encoding context and
    /// rbml encoder. This function will panic if the encoder passed in and the
    /// context encoder are not the same.
    ///
    /// Note that this method is 'practically' safe due to its checking that the
    /// encoder passed in is the same as the one in TLS, but it would still be
    /// possible to construct cases where the EncodingContext is exchanged
    /// while the same encoder is used, thus working with a wrong context.
    pub fn with_encoding_context<'tcx, E, F, R>(encoder: &mut E, f: F) -> R
        where F: FnOnce(&EncodingContext<'tcx>, &mut OpaqueEncoder) -> R,
              E: serialize::Encoder
    {
        unsafe {
            unsafe_with_encoding_context(|ecx, tls_encoder| {
                assert!(encoder as *mut _ as usize == tls_encoder as *mut _ as usize);

                let ecx: &EncodingContext<'tcx> = mem::transmute(ecx);

                f(ecx, tls_encoder)
            })
        }
    }

    /// Execute f with access to the thread-local encoding context and
    /// rbml encoder.
    pub unsafe fn unsafe_with_encoding_context<F, R>(f: F) -> R
        where F: FnOnce(&EncodingContext, &mut OpaqueEncoder) -> R
    {
        TLS_ENCODING.with(|tls| {
            let tls = tls.get().unwrap();
            let tls_payload = tls as *mut (&EncodingContext, &mut OpaqueEncoder);
            f((*tls_payload).0, (*tls_payload).1)
        })
    }

    pub trait DecodingContext<'tcx> {
        fn tcx<'a>(&'a self) -> &'a TyCtxt<'tcx>;
        fn decode_ty(&self, decoder: &mut OpaqueDecoder) -> ty::Ty<'tcx>;
        fn decode_substs(&self, decoder: &mut OpaqueDecoder) -> Substs<'tcx>;
        fn translate_def_id(&self, def_id: DefId) -> DefId;
    }

    thread_local! {
        static TLS_DECODING: Cell<Option<*const TlsPayload>> = Cell::new(None)
    }

    /// Execute f after pushing the given DecodingContext onto the TLS stack.
    pub fn enter_decoding_context<'tcx, F, R>(dcx: &DecodingContext<'tcx>,
                                              decoder: &mut OpaqueDecoder,
                                              f: F) -> R
        where F: FnOnce(&DecodingContext<'tcx>, &mut OpaqueDecoder) -> R
    {
        let tls_payload = (dcx as *const _, decoder as *mut _);
        let tls_ptr = &tls_payload as *const _ as *const TlsPayload;
        TLS_DECODING.with(|tls| {
            let prev = tls.get();
            tls.set(Some(tls_ptr));
            let ret = f(dcx, decoder);
            tls.set(prev);
            return ret
        })
    }

    /// Execute f with access to the thread-local decoding context and
    /// rbml decoder. This function will panic if the decoder passed in and the
    /// context decoder are not the same.
    ///
    /// Note that this method is 'practically' safe due to its checking that the
    /// decoder passed in is the same as the one in TLS, but it would still be
    /// possible to construct cases where the DecodingContext is exchanged
    /// while the same decoder is used, thus working with a wrong context.
    pub fn with_decoding_context<'decoder, 'tcx, D, F, R>(d: &'decoder mut D, f: F) -> R
        where D: serialize::Decoder,
              F: FnOnce(&DecodingContext<'tcx>,
                        &mut OpaqueDecoder) -> R,
              'tcx: 'decoder
    {
        unsafe {
            unsafe_with_decoding_context(|dcx, decoder| {
                assert!((d as *mut _ as usize) == (decoder as *mut _ as usize));

                let dcx: &DecodingContext<'tcx> = mem::transmute(dcx);

                f(dcx, decoder)
            })
        }
    }

    /// Execute f with access to the thread-local decoding context and
    /// rbml decoder.
    pub unsafe fn unsafe_with_decoding_context<F, R>(f: F) -> R
        where F: FnOnce(&DecodingContext, &mut OpaqueDecoder) -> R
    {
        TLS_DECODING.with(|tls| {
            let tls = tls.get().unwrap();
            let tls_payload = tls as *mut (&DecodingContext, &mut OpaqueDecoder);
            f((*tls_payload).0, (*tls_payload).1)
        })
    }
}
