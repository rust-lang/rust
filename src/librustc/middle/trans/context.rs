use core::prelude::*;

use back::{abi, upcall};
use driver::session;
use driver::session::Session;
use lib::llvm::{ModuleRef, ValueRef, TypeRef, BasicBlockRef, BuilderRef};
use lib::llvm::{ContextRef, True, False, Bool};
use lib::llvm::{llvm, TargetData, TypeNames, associate_type, name_has_type};
use lib;
use metadata::common::LinkMeta;
use middle::astencode;
use middle::resolve;
use middle::trans::adt;
use middle::trans::base;
use middle::trans::build;
use middle::trans::datum;
use middle::trans::debuginfo;
use middle::trans::glue;
use middle::trans::reachable;
use middle::trans::shape;
use middle::trans::type_of;
use middle::trans::type_use;
use middle::trans::write_guard;
use middle::ty::substs;
use middle::ty;
use middle::typeck;
use middle::borrowck::root_map_key;

use core::cast::transmute;
use core::cast;
use core::hash;
use core::hashmap::{HashMap, HashSet};
use core::libc::{c_uint, c_longlong, c_ulonglong};
use core::str;
use core::to_bytes;
use core::vec::raw::to_ptr;
use core::vec;
use syntax::ast::ident;
use syntax::ast_map::{path, path_elt};
use syntax::codemap::span;
use syntax::parse::token;
use syntax::{ast, ast_map};
use syntax::abi::{X86, X86_64, Arm, Mips};

use middle::trans::common::{ExternMap,tydesc_info,BuilderRef_res,Stats,namegen,addrspace_gen};
use middle::trans::common::{mono_id};

pub struct CrateContext {
     sess: session::Session,
     llmod: ModuleRef,
     llcx: ContextRef,
     td: TargetData,
     tn: @TypeNames,
     externs: ExternMap,
     intrinsics: HashMap<&'static str, ValueRef>,
     item_vals: @mut HashMap<ast::node_id, ValueRef>,
     exp_map2: resolve::ExportMap2,
     reachable: reachable::map,
     item_symbols: @mut HashMap<ast::node_id, ~str>,
     link_meta: LinkMeta,
     enum_sizes: @mut HashMap<ty::t, uint>,
     discrims: @mut HashMap<ast::def_id, ValueRef>,
     discrim_symbols: @mut HashMap<ast::node_id, @str>,
     tydescs: @mut HashMap<ty::t, @mut tydesc_info>,
     // Set when running emit_tydescs to enforce that no more tydescs are
     // created.
     finished_tydescs: @mut bool,
     // Track mapping of external ids to local items imported for inlining
     external: @mut HashMap<ast::def_id, Option<ast::node_id>>,
     // Cache instances of monomorphized functions
     monomorphized: @mut HashMap<mono_id, ValueRef>,
     monomorphizing: @mut HashMap<ast::def_id, uint>,
     // Cache computed type parameter uses (see type_use.rs)
     type_use_cache: @mut HashMap<ast::def_id, @~[type_use::type_uses]>,
     // Cache generated vtables
     vtables: @mut HashMap<mono_id, ValueRef>,
     // Cache of constant strings,
     const_cstr_cache: @mut HashMap<@str, ValueRef>,

     // Reverse-direction for const ptrs cast from globals.
     // Key is an int, cast from a ValueRef holding a *T,
     // Val is a ValueRef holding a *[T].
     //
     // Needed because LLVM loses pointer->pointee association
     // when we ptrcast, and we have to ptrcast during translation
     // of a [T] const because we form a slice, a [*T,int] pair, not
     // a pointer to an LLVM array type.
     const_globals: @mut HashMap<int, ValueRef>,

     // Cache of emitted const values
     const_values: @mut HashMap<ast::node_id, ValueRef>,

     // Cache of external const values
     extern_const_values: @mut HashMap<ast::def_id, ValueRef>,

     module_data: @mut HashMap<~str, ValueRef>,
     lltypes: @mut HashMap<ty::t, TypeRef>,
     llsizingtypes: @mut HashMap<ty::t, TypeRef>,
     adt_reprs: @mut HashMap<ty::t, @adt::Repr>,
     names: namegen,
     next_addrspace: addrspace_gen,
     symbol_hasher: @mut hash::State,
     type_hashcodes: @mut HashMap<ty::t, @str>,
     type_short_names: @mut HashMap<ty::t, ~str>,
     all_llvm_symbols: @mut HashSet<@str>,
     tcx: ty::ctxt,
     maps: astencode::Maps,
     stats: @mut Stats,
     upcalls: @upcall::Upcalls,
     tydesc_type: TypeRef,
     int_type: TypeRef,
     float_type: TypeRef,
     opaque_vec_type: TypeRef,
     builder: BuilderRef_res,
     shape_cx: shape::Ctxt,
     crate_map: ValueRef,
     // Set when at least one function uses GC. Needed so that
     // decl_gc_metadata knows whether to link to the module metadata, which
     // is not emitted by LLVM's GC pass when no functions use GC.
     uses_gc: @mut bool,
     dbg_cx: Option<debuginfo::DebugContext>,
     do_not_commit_warning_issued: @mut bool
}
