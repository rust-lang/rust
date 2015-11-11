// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// See doc.rs for documentation.
mod doc;

use self::VariableAccess::*;
use self::VariableKind::*;

use self::utils::{DIB, span_start, assert_type_for_node_id, contains_nodebug_attribute,
                  create_DIArray, is_node_local_to_unit};
use self::namespace::{namespace_for_item, NamespaceTreeNode};
use self::type_names::compute_debuginfo_type_name;
use self::metadata::{type_metadata, diverging_type_metadata};
use self::metadata::{file_metadata, scope_metadata, TypeMap, compile_unit_metadata};
use self::source_loc::InternalDebugLocation;

use llvm;
use llvm::{ModuleRef, ContextRef, ValueRef};
use llvm::debuginfo::{DIFile, DIType, DIScope, DIBuilderRef, DISubprogram, DIArray,
                      DIDescriptor, FlagPrototyped};
use middle::def_id::DefId;
use middle::infer::normalize_associated_type;
use middle::subst::{self, Substs};
use rustc_front;
use rustc_front::hir;

use trans::common::{NodeIdAndSpan, CrateContext, FunctionContext, Block};
use trans;
use trans::{monomorphize, type_of};
use middle::infer;
use middle::ty::{self, Ty};
use session::config::{self, FullDebugInfo, LimitedDebugInfo, NoDebugInfo};
use util::nodemap::{NodeMap, FnvHashMap, FnvHashSet};
use rustc::front::map as hir_map;

use libc::c_uint;
use std::cell::{Cell, RefCell};
use std::ffi::CString;
use std::ptr;
use std::rc::Rc;

use syntax::codemap::{Span, Pos};
use syntax::{abi, ast, codemap};
use syntax::attr::IntType;
use syntax::parse::token::{self, special_idents};

pub mod gdb;
mod utils;
mod namespace;
mod type_names;
mod metadata;
mod create_scope_map;
mod source_loc;

pub use self::source_loc::set_source_location;
pub use self::source_loc::clear_source_location;
pub use self::source_loc::start_emitting_source_locations;
pub use self::source_loc::get_cleanup_debug_loc_for_ast_node;
pub use self::source_loc::with_source_location_override;
pub use self::metadata::create_match_binding_metadata;
pub use self::metadata::create_argument_metadata;
pub use self::metadata::create_captured_var_metadata;
pub use self::metadata::create_global_var_metadata;
pub use self::metadata::create_local_var_metadata;

#[allow(non_upper_case_globals)]
const DW_TAG_auto_variable: c_uint = 0x100;
#[allow(non_upper_case_globals)]
const DW_TAG_arg_variable: c_uint = 0x101;

/// A context object for maintaining all state needed by the debuginfo module.
pub struct CrateDebugContext<'tcx> {
    llcontext: ContextRef,
    builder: DIBuilderRef,
    current_debug_location: Cell<InternalDebugLocation>,
    created_files: RefCell<FnvHashMap<String, DIFile>>,
    created_enum_disr_types: RefCell<FnvHashMap<(DefId, IntType), DIType>>,

    type_map: RefCell<TypeMap<'tcx>>,
    namespace_map: RefCell<FnvHashMap<Vec<ast::Name>, Rc<NamespaceTreeNode>>>,

    // This collection is used to assert that composite types (structs, enums,
    // ...) have their members only set once:
    composite_types_completed: RefCell<FnvHashSet<DIType>>,
}

impl<'tcx> CrateDebugContext<'tcx> {
    pub fn new(llmod: ModuleRef) -> CrateDebugContext<'tcx> {
        debug!("CrateDebugContext::new");
        let builder = unsafe { llvm::LLVMDIBuilderCreate(llmod) };
        // DIBuilder inherits context from the module, so we'd better use the same one
        let llcontext = unsafe { llvm::LLVMGetModuleContext(llmod) };
        return CrateDebugContext {
            llcontext: llcontext,
            builder: builder,
            current_debug_location: Cell::new(InternalDebugLocation::UnknownLocation),
            created_files: RefCell::new(FnvHashMap()),
            created_enum_disr_types: RefCell::new(FnvHashMap()),
            type_map: RefCell::new(TypeMap::new()),
            namespace_map: RefCell::new(FnvHashMap()),
            composite_types_completed: RefCell::new(FnvHashSet()),
        };
    }
}

pub enum FunctionDebugContext {
    RegularContext(Box<FunctionDebugContextData>),
    DebugInfoDisabled,
    FunctionWithoutDebugInfo,
}

impl FunctionDebugContext {
    fn get_ref<'a>(&'a self,
                   cx: &CrateContext,
                   span: Span)
                   -> &'a FunctionDebugContextData {
        match *self {
            FunctionDebugContext::RegularContext(box ref data) => data,
            FunctionDebugContext::DebugInfoDisabled => {
                cx.sess().span_bug(span,
                                   FunctionDebugContext::debuginfo_disabled_message());
            }
            FunctionDebugContext::FunctionWithoutDebugInfo => {
                cx.sess().span_bug(span,
                                   FunctionDebugContext::should_be_ignored_message());
            }
        }
    }

    fn debuginfo_disabled_message() -> &'static str {
        "debuginfo: Error trying to access FunctionDebugContext although debug info is disabled!"
    }

    fn should_be_ignored_message() -> &'static str {
        "debuginfo: Error trying to access FunctionDebugContext for function that should be \
         ignored by debug info!"
    }
}

struct FunctionDebugContextData {
    scope_map: RefCell<NodeMap<DIScope>>,
    fn_metadata: DISubprogram,
    argument_counter: Cell<usize>,
    source_locations_enabled: Cell<bool>,
    source_location_override: Cell<bool>,
}

pub enum VariableAccess<'a> {
    // The llptr given is an alloca containing the variable's value
    DirectVariable { alloca: ValueRef },
    // The llptr given is an alloca containing the start of some pointer chain
    // leading to the variable's content.
    IndirectVariable { alloca: ValueRef, address_operations: &'a [i64] }
}

pub enum VariableKind {
    ArgumentVariable(usize /*index*/),
    LocalVariable,
    CapturedVariable,
}

/// Create any deferred debug metadata nodes
pub fn finalize(cx: &CrateContext) {
    if cx.dbg_cx().is_none() {
        return;
    }

    debug!("finalize");
    let _ = compile_unit_metadata(cx);

    if gdb::needs_gdb_debug_scripts_section(cx) {
        // Add a .debug_gdb_scripts section to this compile-unit. This will
        // cause GDB to try and load the gdb_load_rust_pretty_printers.py file,
        // which activates the Rust pretty printers for binary this section is
        // contained in.
        gdb::get_or_insert_gdb_debug_scripts_section_global(cx);
    }

    unsafe {
        llvm::LLVMDIBuilderFinalize(DIB(cx));
        llvm::LLVMDIBuilderDispose(DIB(cx));
        // Debuginfo generation in LLVM by default uses a higher
        // version of dwarf than OS X currently understands. We can
        // instruct LLVM to emit an older version of dwarf, however,
        // for OS X to understand. For more info see #11352
        // This can be overridden using --llvm-opts -dwarf-version,N.
        // Android has the same issue (#22398)
        if cx.sess().target.target.options.is_like_osx ||
           cx.sess().target.target.options.is_like_android {
            llvm::LLVMRustAddModuleFlag(cx.llmod(),
                                        "Dwarf Version\0".as_ptr() as *const _,
                                        2)
        }

        // Prevent bitcode readers from deleting the debug info.
        let ptr = "Debug Info Version\0".as_ptr();
        llvm::LLVMRustAddModuleFlag(cx.llmod(), ptr as *const _,
                                    llvm::LLVMRustDebugMetadataVersion());
    };
}

/// Creates the function-specific debug context.
///
/// Returns the FunctionDebugContext for the function which holds state needed
/// for debug info creation. The function may also return another variant of the
/// FunctionDebugContext enum which indicates why no debuginfo should be created
/// for the function.
pub fn create_function_debug_context<'a, 'tcx>(cx: &CrateContext<'a, 'tcx>,
                                               fn_ast_id: ast::NodeId,
                                               param_substs: &Substs<'tcx>,
                                               llfn: ValueRef) -> FunctionDebugContext {
    if cx.sess().opts.debuginfo == NoDebugInfo {
        return FunctionDebugContext::DebugInfoDisabled;
    }

    // Clear the debug location so we don't assign them in the function prelude.
    // Do this here already, in case we do an early exit from this function.
    source_loc::set_debug_location(cx, InternalDebugLocation::UnknownLocation);

    if fn_ast_id == ast::DUMMY_NODE_ID {
        // This is a function not linked to any source location, so don't
        // generate debuginfo for it.
        return FunctionDebugContext::FunctionWithoutDebugInfo;
    }

    let empty_generics = rustc_front::util::empty_generics();

    let fnitem = cx.tcx().map.get(fn_ast_id);

    let (name, fn_decl, generics, top_level_block, span, has_path) = match fnitem {
        hir_map::NodeItem(ref item) => {
            if contains_nodebug_attribute(&item.attrs) {
                return FunctionDebugContext::FunctionWithoutDebugInfo;
            }

            match item.node {
                hir::ItemFn(ref fn_decl, _, _, _, ref generics, ref top_level_block) => {
                    (item.name, fn_decl, generics, top_level_block, item.span, true)
                }
                _ => {
                    cx.sess().span_bug(item.span,
                        "create_function_debug_context: item bound to non-function");
                }
            }
        }
        hir_map::NodeImplItem(impl_item) => {
            match impl_item.node {
                hir::ImplItem_::Method(ref sig, ref body) => {
                    if contains_nodebug_attribute(&impl_item.attrs) {
                        return FunctionDebugContext::FunctionWithoutDebugInfo;
                    }

                    (impl_item.name,
                     &sig.decl,
                     &sig.generics,
                     body,
                     impl_item.span,
                     true)
                }
                _ => {
                    cx.sess().span_bug(impl_item.span,
                                       "create_function_debug_context() \
                                        called on non-method impl item?!")
                }
            }
        }
        hir_map::NodeExpr(ref expr) => {
            match expr.node {
                hir::ExprClosure(_, ref fn_decl, ref top_level_block) => {
                    let name = format!("fn{}", token::gensym("fn"));
                    let name = token::intern(&name[..]);
                    (name, fn_decl,
                        // This is not quite right. It should actually inherit
                        // the generics of the enclosing function.
                        &empty_generics,
                        top_level_block,
                        expr.span,
                        // Don't try to lookup the item path:
                        false)
                }
                _ => cx.sess().span_bug(expr.span,
                        "create_function_debug_context: expected an expr_fn_block here")
            }
        }
        hir_map::NodeTraitItem(trait_item) => {
            match trait_item.node {
                hir::MethodTraitItem(ref sig, Some(ref body)) => {
                    if contains_nodebug_attribute(&trait_item.attrs) {
                        return FunctionDebugContext::FunctionWithoutDebugInfo;
                    }

                    (trait_item.name,
                     &sig.decl,
                     &sig.generics,
                     body,
                     trait_item.span,
                     true)
                }
                _ => {
                    cx.sess()
                      .bug(&format!("create_function_debug_context: \
                                    unexpected sort of node: {:?}",
                                    fnitem))
                }
            }
        }
        hir_map::NodeForeignItem(..) |
        hir_map::NodeVariant(..) |
        hir_map::NodeStructCtor(..) => {
            return FunctionDebugContext::FunctionWithoutDebugInfo;
        }
        _ => cx.sess().bug(&format!("create_function_debug_context: \
                                    unexpected sort of node: {:?}",
                                   fnitem))
    };

    // This can be the case for functions inlined from another crate
    if span == codemap::DUMMY_SP {
        return FunctionDebugContext::FunctionWithoutDebugInfo;
    }

    let loc = span_start(cx, span);
    let file_metadata = file_metadata(cx, &loc.file.name);

    let function_type_metadata = unsafe {
        let fn_signature = get_function_signature(cx,
                                                  fn_ast_id,
                                                  param_substs,
                                                  span);
        llvm::LLVMDIBuilderCreateSubroutineType(DIB(cx), file_metadata, fn_signature)
    };

    // Get_template_parameters() will append a `<...>` clause to the function
    // name if necessary.
    let mut function_name = name.to_string();
    let template_parameters = get_template_parameters(cx,
                                                      generics,
                                                      param_substs,
                                                      file_metadata,
                                                      &mut function_name);

    // There is no hir_map::Path for hir::ExprClosure-type functions. For now,
    // just don't put them into a namespace. In the future this could be improved
    // somehow (storing a path in the hir_map, or construct a path using the
    // enclosing function).
    let (linkage_name, containing_scope) = if has_path {
        let fn_ast_def_id = cx.tcx().map.local_def_id(fn_ast_id);
        let namespace_node = namespace_for_item(cx, fn_ast_def_id);
        let linkage_name = namespace_node.mangled_name_of_contained_item(
            &function_name[..]);
        let containing_scope = namespace_node.scope;
        (linkage_name, containing_scope)
    } else {
        (function_name.clone(), file_metadata)
    };

    // Clang sets this parameter to the opening brace of the function's block,
    // so let's do this too.
    let scope_line = span_start(cx, top_level_block.span).line;

    let is_local_to_unit = is_node_local_to_unit(cx, fn_ast_id);

    let function_name = CString::new(function_name).unwrap();
    let linkage_name = CString::new(linkage_name).unwrap();
    let fn_metadata = unsafe {
        llvm::LLVMDIBuilderCreateFunction(
            DIB(cx),
            containing_scope,
            function_name.as_ptr(),
            linkage_name.as_ptr(),
            file_metadata,
            loc.line as c_uint,
            function_type_metadata,
            is_local_to_unit,
            true,
            scope_line as c_uint,
            FlagPrototyped as c_uint,
            cx.sess().opts.optimize != config::No,
            llfn,
            template_parameters,
            ptr::null_mut())
    };

    let scope_map = create_scope_map::create_scope_map(cx,
                                                       &fn_decl.inputs,
                                                       &*top_level_block,
                                                       fn_metadata,
                                                       fn_ast_id);

    // Initialize fn debug context (including scope map and namespace map)
    let fn_debug_context = box FunctionDebugContextData {
        scope_map: RefCell::new(scope_map),
        fn_metadata: fn_metadata,
        argument_counter: Cell::new(1),
        source_locations_enabled: Cell::new(false),
        source_location_override: Cell::new(false),
    };



    return FunctionDebugContext::RegularContext(fn_debug_context);

    fn get_function_signature<'a, 'tcx>(cx: &CrateContext<'a, 'tcx>,
                                        fn_ast_id: ast::NodeId,
                                        param_substs: &Substs<'tcx>,
                                        error_reporting_span: Span) -> DIArray {
        if cx.sess().opts.debuginfo == LimitedDebugInfo {
            return create_DIArray(DIB(cx), &[]);
        }

        // Return type -- llvm::DIBuilder wants this at index 0
        assert_type_for_node_id(cx, fn_ast_id, error_reporting_span);
        let fn_type = cx.tcx().node_id_to_type(fn_ast_id);
        let fn_type = monomorphize::apply_param_substs(cx.tcx(), param_substs, &fn_type);

        let (sig, abi) = match fn_type.sty {
            ty::TyBareFn(_, ref barefnty) => {
                let sig = cx.tcx().erase_late_bound_regions(&barefnty.sig);
                let sig = infer::normalize_associated_type(cx.tcx(), &sig);
                (sig, barefnty.abi)
            }
            ty::TyClosure(def_id, ref substs) => {
                let closure_type = cx.tcx().closure_type(def_id, substs);
                let sig = cx.tcx().erase_late_bound_regions(&closure_type.sig);
                let sig = infer::normalize_associated_type(cx.tcx(), &sig);
                (sig, closure_type.abi)
            }

            _ => cx.sess().bug("get_function_metdata: Expected a function type!")
        };

        let mut signature = Vec::with_capacity(sig.inputs.len() + 1);

        // Return type -- llvm::DIBuilder wants this at index 0
        signature.push(match sig.output {
            ty::FnConverging(ret_ty) => match ret_ty.sty {
                ty::TyTuple(ref tys) if tys.is_empty() => ptr::null_mut(),
                _ => type_metadata(cx, ret_ty, codemap::DUMMY_SP)
            },
            ty::FnDiverging => diverging_type_metadata(cx)
        });

        let inputs = &if abi == abi::RustCall {
            type_of::untuple_arguments(cx, &sig.inputs)
        } else {
            sig.inputs
        };

        // Arguments types
        for &argument_type in inputs {
            signature.push(type_metadata(cx, argument_type, codemap::DUMMY_SP));
        }

        return create_DIArray(DIB(cx), &signature[..]);
    }

    fn get_template_parameters<'a, 'tcx>(cx: &CrateContext<'a, 'tcx>,
                                         generics: &hir::Generics,
                                         param_substs: &Substs<'tcx>,
                                         file_metadata: DIFile,
                                         name_to_append_suffix_to: &mut String)
                                         -> DIArray
    {
        let self_type = param_substs.self_ty();
        let self_type = normalize_associated_type(cx.tcx(), &self_type);

        // Only true for static default methods:
        let has_self_type = self_type.is_some();

        if !generics.is_type_parameterized() && !has_self_type {
            return create_DIArray(DIB(cx), &[]);
        }

        name_to_append_suffix_to.push('<');

        // The list to be filled with template parameters:
        let mut template_params: Vec<DIDescriptor> =
            Vec::with_capacity(generics.ty_params.len() + 1);

        // Handle self type
        if has_self_type {
            let actual_self_type = self_type.unwrap();
            // Add self type name to <...> clause of function name
            let actual_self_type_name = compute_debuginfo_type_name(
                cx,
                actual_self_type,
                true);

            name_to_append_suffix_to.push_str(&actual_self_type_name[..]);

            if generics.is_type_parameterized() {
                name_to_append_suffix_to.push_str(",");
            }

            // Only create type information if full debuginfo is enabled
            if cx.sess().opts.debuginfo == FullDebugInfo {
                let actual_self_type_metadata = type_metadata(cx,
                                                              actual_self_type,
                                                              codemap::DUMMY_SP);

                let name = special_idents::type_self.name.as_str();

                let name = CString::new(name.as_bytes()).unwrap();
                let param_metadata = unsafe {
                    llvm::LLVMDIBuilderCreateTemplateTypeParameter(
                        DIB(cx),
                        ptr::null_mut(),
                        name.as_ptr(),
                        actual_self_type_metadata,
                        file_metadata,
                        0,
                        0)
                };

                template_params.push(param_metadata);
            }
        }

        // Handle other generic parameters
        let actual_types = param_substs.types.get_slice(subst::FnSpace);
        for (index, &hir::TyParam{ name, .. }) in generics.ty_params.iter().enumerate() {
            let actual_type = actual_types[index];
            // Add actual type name to <...> clause of function name
            let actual_type_name = compute_debuginfo_type_name(cx,
                                                               actual_type,
                                                               true);
            name_to_append_suffix_to.push_str(&actual_type_name[..]);

            if index != generics.ty_params.len() - 1 {
                name_to_append_suffix_to.push_str(",");
            }

            // Again, only create type information if full debuginfo is enabled
            if cx.sess().opts.debuginfo == FullDebugInfo {
                let actual_type_metadata = type_metadata(cx, actual_type, codemap::DUMMY_SP);
                let name = CString::new(name.as_str().as_bytes()).unwrap();
                let param_metadata = unsafe {
                    llvm::LLVMDIBuilderCreateTemplateTypeParameter(
                        DIB(cx),
                        ptr::null_mut(),
                        name.as_ptr(),
                        actual_type_metadata,
                        file_metadata,
                        0,
                        0)
                };
                template_params.push(param_metadata);
            }
        }

        name_to_append_suffix_to.push('>');

        return create_DIArray(DIB(cx), &template_params[..]);
    }
}

fn declare_local<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                             variable_name: ast::Name,
                             variable_type: Ty<'tcx>,
                             scope_metadata: DIScope,
                             variable_access: VariableAccess,
                             variable_kind: VariableKind,
                             span: Span) {
    let cx: &CrateContext = bcx.ccx();

    let filename = span_start(cx, span).file.name.clone();
    let file_metadata = file_metadata(cx, &filename[..]);

    let loc = span_start(cx, span);
    let type_metadata = type_metadata(cx, variable_type, span);

    let (argument_index, dwarf_tag) = match variable_kind {
        ArgumentVariable(index) => (index as c_uint, DW_TAG_arg_variable),
        LocalVariable    |
        CapturedVariable => (0, DW_TAG_auto_variable)
    };

    let name = CString::new(variable_name.as_str().as_bytes()).unwrap();
    match (variable_access, &[][..]) {
        (DirectVariable { alloca }, address_operations) |
        (IndirectVariable {alloca, address_operations}, _) => {
            let metadata = unsafe {
                llvm::LLVMDIBuilderCreateVariable(
                    DIB(cx),
                    dwarf_tag,
                    scope_metadata,
                    name.as_ptr(),
                    file_metadata,
                    loc.line as c_uint,
                    type_metadata,
                    cx.sess().opts.optimize != config::No,
                    0,
                    address_operations.as_ptr(),
                    address_operations.len() as c_uint,
                    argument_index)
            };
            source_loc::set_debug_location(cx, InternalDebugLocation::new(scope_metadata,
                                                                          loc.line,
                                                                          loc.col.to_usize()));
            unsafe {
                let debug_loc = llvm::LLVMGetCurrentDebugLocation(cx.raw_builder());
                let instr = llvm::LLVMDIBuilderInsertDeclareAtEnd(
                    DIB(cx),
                    alloca,
                    metadata,
                    address_operations.as_ptr(),
                    address_operations.len() as c_uint,
                    debug_loc,
                    bcx.llbb);

                llvm::LLVMSetInstDebugLocation(trans::build::B(bcx).llbuilder, instr);
            }
        }
    }

    match variable_kind {
        ArgumentVariable(_) | CapturedVariable => {
            assert!(!bcx.fcx
                        .debug_context
                        .get_ref(cx, span)
                        .source_locations_enabled
                        .get());
            source_loc::set_debug_location(cx, InternalDebugLocation::UnknownLocation);
        }
        _ => { /* nothing to do */ }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum DebugLoc {
    At(ast::NodeId, Span),
    None
}

impl DebugLoc {
    pub fn apply(&self, fcx: &FunctionContext) {
        match *self {
            DebugLoc::At(node_id, span) => {
                source_loc::set_source_location(fcx, node_id, span);
            }
            DebugLoc::None => {
                source_loc::clear_source_location(fcx);
            }
        }
    }
}

pub trait ToDebugLoc {
    fn debug_loc(&self) -> DebugLoc;
}

impl ToDebugLoc for hir::Expr {
    fn debug_loc(&self) -> DebugLoc {
        DebugLoc::At(self.id, self.span)
    }
}

impl ToDebugLoc for NodeIdAndSpan {
    fn debug_loc(&self) -> DebugLoc {
        DebugLoc::At(self.id, self.span)
    }
}

impl ToDebugLoc for Option<NodeIdAndSpan> {
    fn debug_loc(&self) -> DebugLoc {
        match *self {
            Some(NodeIdAndSpan { id, span }) => DebugLoc::At(id, span),
            None => DebugLoc::None
        }
    }
}
