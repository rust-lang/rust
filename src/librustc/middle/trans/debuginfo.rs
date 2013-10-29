// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!
# Debug Info Module

This module serves the purpose of generating debug symbols. We use LLVM's
[source level debugging](http://llvm.org/docs/SourceLevelDebugging.html) features for generating
the debug information. The general principle is this:

Given the right metadata in the LLVM IR, the LLVM code generator is able to create DWARF debug
symbols for the given code. The [metadata](http://llvm.org/docs/LangRef.html#metadata-type) is
structured much like DWARF *debugging information entries* (DIE), representing type information
such as datatype layout, function signatures, block layout, variable location and scope information,
etc. It is the purpose of this module to generate correct metadata and insert it into the LLVM IR.

As the exact format of metadata trees may change between different LLVM versions, we now use LLVM
[DIBuilder](http://llvm.org/docs/doxygen/html/classllvm_1_1DIBuilder.html) to create metadata
where possible. This will hopefully ease the adaption of this module to future LLVM versions.

The public API of the module is a set of functions that will insert the correct metadata into the
LLVM IR when called with the right parameters. The module is thus driven from an outside client with
functions like `debuginfo::create_local_var_metadata(bcx: block, local: &ast::local)`.

Internally the module will try to reuse already created metadata by utilizing a cache. The way to
get a shared metadata node when needed is thus to just call the corresponding function in this
module:

    let file_metadata = file_metadata(crate_context, path);

The function will take care of probing the cache for an existing node for that exact file path.

All private state used by the module is stored within either the CrateDebugContext struct (owned by
the CrateContext) or the FunctionDebugContext (owned by the FunctionContext).

This file consists of three conceptual sections:
1. The public interface of the module
2. Module-internal metadata creation functions
3. Minor utility functions


## Recursive Types
Some kinds of types, such as structs and enums can be recursive. That means that the type definition
of some type X refers to some other type which in turn (transitively) refers to X. This introduces
cycles into the type referral graph. A naive algorithm doing an on-demand, depth-first traversal of
this graph when describing types, can get trapped in an endless loop when it reaches such a cycle.

For example, the following simple type for a singly-linked list...

```
struct List {
    value: int,
    tail: Option<@List>,
}
```

will generate the following callstack with a naive DFS algorithm:

```
describe(t = List)
  describe(t = int)
  describe(t = Option<@List>)
    describe(t = @List)
      describe(t = List) // at the beginning again...
      ...
```

To break cycles like these, we use "forward declarations". That is, when the algorithm encounters a
possibly recursive type (any struct or enum), it immediately creates a type description node and
inserts it into the cache *before* describing the members of the type. This type description is just
a stub (as type members are not described and added to it yet) but it allows the algorithm to
already refer to the type. After the stub is inserted into the cache, the algorithm continues as
before. If it now encounters a recursive reference, it will hit the cache and does not try to
describe the type anew.

This behaviour is encapsulated in the 'RecursiveTypeDescription' enum, which represents a kind of
continuation, storing all state needed to continue traversal at the type members after the type has
been registered with the cache. (This implementation approach might be a tad over-engineered and
may change in the future)

*/


use driver::session;
use lib::llvm::llvm;
use lib::llvm::{ModuleRef, ContextRef, ValueRef};
use lib::llvm::debuginfo::*;
use middle::trans::common::*;
use middle::trans::machine;
use middle::trans::type_of;
use middle::trans::type_::Type;
use middle::trans::adt;
use middle::trans;
use middle::ty;
use middle::pat_util;
use util::ppaux;

use std::c_str::ToCStr;
use std::hashmap::HashMap;
use std::hashmap::HashSet;
use std::libc::{c_uint, c_ulonglong, c_longlong};
use std::ptr;
use std::unstable::atomics;
use std::vec;
use syntax::codemap::{Span, Pos};
use syntax::{ast, codemap, ast_util, ast_map, opt_vec};
use syntax::parse::token;
use syntax::parse::token::special_idents;

static DW_LANG_RUST: c_uint = 0x9000;

static DW_TAG_auto_variable: c_uint = 0x100;
static DW_TAG_arg_variable: c_uint = 0x101;

static DW_ATE_boolean: c_uint = 0x02;
static DW_ATE_float: c_uint = 0x04;
static DW_ATE_signed: c_uint = 0x05;
static DW_ATE_signed_char: c_uint = 0x06;
static DW_ATE_unsigned: c_uint = 0x07;
static DW_ATE_unsigned_char: c_uint = 0x08;

//=-------------------------------------------------------------------------------------------------
//  Public Interface of debuginfo module
//=-------------------------------------------------------------------------------------------------

/// A context object for maintaining all state needed by the debuginfo module.
pub struct CrateDebugContext {
    priv crate_file: ~str,
    priv llcontext: ContextRef,
    priv builder: DIBuilderRef,
    priv current_debug_location: DebugLocation,
    priv created_files: HashMap<~str, DIFile>,
    priv created_types: HashMap<uint, DIType>,
    priv namespace_map: HashMap<~[ast::Ident], @NamespaceTreeNode>,
    // This collection is used to assert that composite types (structs, enums, ...) have their
    // members only set once:
    priv composite_types_completed: HashSet<DIType>,
}

impl CrateDebugContext {
    pub fn new(llmod: ModuleRef, crate: ~str) -> CrateDebugContext {
        debug!("CrateDebugContext::new");
        let builder = unsafe { llvm::LLVMDIBuilderCreate(llmod) };
        // DIBuilder inherits context from the module, so we'd better use the same one
        let llcontext = unsafe { llvm::LLVMGetModuleContext(llmod) };
        return CrateDebugContext {
            crate_file: crate,
            llcontext: llcontext,
            builder: builder,
            current_debug_location: UnknownLocation,
            created_files: HashMap::new(),
            created_types: HashMap::new(),
            namespace_map: HashMap::new(),
            composite_types_completed: HashSet::new(),
        };
    }
}

pub enum FunctionDebugContext {
    priv FunctionDebugContext(~FunctionDebugContextData),
    priv DebugInfoDisabled,
    priv FunctionWithoutDebugInfo,
}

impl FunctionDebugContext {
    fn get_ref<'a>(&'a self, cx: &CrateContext, span: Span) -> &'a FunctionDebugContextData {
        match *self {
            FunctionDebugContext(~ref data) => data,
            DebugInfoDisabled => {
                cx.sess.span_bug(span, FunctionDebugContext::debuginfo_disabled_message());
            }
            FunctionWithoutDebugInfo => {
                cx.sess.span_bug(span, FunctionDebugContext::should_be_ignored_message());
            }
        }
    }

    fn get_mut_ref<'a>(&'a mut self,
                       cx: &CrateContext,
                       span: Span)
                    -> &'a mut FunctionDebugContextData {
        match *self {
            FunctionDebugContext(~ref mut data) => data,
            DebugInfoDisabled => {
                cx.sess.span_bug(span, FunctionDebugContext::debuginfo_disabled_message());
            }
            FunctionWithoutDebugInfo => {
                cx.sess.span_bug(span, FunctionDebugContext::should_be_ignored_message());
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
    scope_map: HashMap<ast::NodeId, DIScope>,
    fn_metadata: DISubprogram,
    argument_counter: uint,
    source_locations_enabled: bool,
}

enum VariableAccess<'self> {
    // The llptr given is an alloca containing the variable's value
    DirectVariable { alloca: ValueRef },
    // The llptr given is an alloca containing the start of some pointer chain leading to the
    // variable's content.
    IndirectVariable { alloca: ValueRef, address_operations: &'self [ValueRef] }
}

enum VariableKind {
    ArgumentVariable(uint /*index*/),
    LocalVariable,
    CapturedVariable,
}

/// Create any deferred debug metadata nodes
pub fn finalize(cx: @mut CrateContext) {
    if cx.dbg_cx.is_none() {
        return;
    }

    debug!("finalize");
    compile_unit_metadata(cx);
    unsafe {
        llvm::LLVMDIBuilderFinalize(DIB(cx));
        llvm::LLVMDIBuilderDispose(DIB(cx));
    };
}

/// Creates debug information for the given local variable.
///
/// Adds the created metadata nodes directly to the crate's IR.
pub fn create_local_var_metadata(bcx: @mut Block,
                                 local: &ast::Local) {
    if fn_should_be_ignored(bcx.fcx) {
        return;
    }

    let cx = bcx.ccx();
    let def_map = cx.tcx.def_map;

    do pat_util::pat_bindings(def_map, local.pat) |_, node_id, span, path_ref| {

        let var_ident = ast_util::path_to_ident(path_ref);
        let var_type = node_id_type(bcx, node_id);

        let llptr = match bcx.fcx.lllocals.find_copy(&node_id) {
            Some(v) => v,
            None => {
                bcx.tcx().sess.span_bug(span,
                    format!("No entry in lllocals table for {:?}", node_id));
            }
        };

        let scope_metadata = scope_metadata(bcx.fcx, node_id, span);

        declare_local(bcx,
                      var_ident,
                      var_type,
                      scope_metadata,
                      DirectVariable { alloca: llptr },
                      LocalVariable,
                      span);
    }
}

/// Creates debug information for a variable captured in a closure.
///
/// Adds the created metadata nodes directly to the crate's IR.
pub fn create_captured_var_metadata(bcx: @mut Block,
                                    node_id: ast::NodeId,
                                    env_data_type: ty::t,
                                    env_pointer: ValueRef,
                                    env_index: uint,
                                    closure_sigil: ast::Sigil,
                                    span: Span) {
    if fn_should_be_ignored(bcx.fcx) {
        return;
    }

    let cx = bcx.ccx();

    let ast_item = cx.tcx.items.find_copy(&node_id);
    let variable_ident = match ast_item {
        None => {
            cx.sess.span_bug(span, "debuginfo::create_captured_var_metadata() - NodeId not found");
        }
        Some(ast_map::node_local(ident)) => ident,
        Some(ast_map::node_arg(@ast::Pat { node: ast::PatIdent(_, ref path, _), _ })) => {
            ast_util::path_to_ident(path)
        }
        _ => {
            cx.sess.span_bug(span, format!("debuginfo::create_captured_var_metadata() - \
                Captured var-id refers to unexpected ast_map variant: {:?}", ast_item));
        }
    };

    let variable_type = node_id_type(bcx, node_id);
    let scope_metadata = bcx.fcx.debug_context.get_ref(cx, span).fn_metadata;

    let llvm_env_data_type = type_of::type_of(cx, env_data_type);
    let byte_offset_of_var_in_env = machine::llelement_offset(cx, llvm_env_data_type, env_index);

    let address_operations = unsafe {
        [llvm::LLVMDIBuilderCreateOpDeref(Type::i64().to_ref()),
         llvm::LLVMDIBuilderCreateOpPlus(Type::i64().to_ref()),
         C_i64(byte_offset_of_var_in_env as i64),
         llvm::LLVMDIBuilderCreateOpDeref(Type::i64().to_ref())]
    };

    let address_op_count = match closure_sigil {
        ast::BorrowedSigil => {
            address_operations.len()
        }
        ast::ManagedSigil | ast::OwnedSigil => {
            address_operations.len() - 1
        }
    };

    let variable_access = IndirectVariable {
        alloca: env_pointer,
        address_operations: address_operations.slice_to(address_op_count)
    };

    declare_local(bcx,
                  variable_ident,
                  variable_type,
                  scope_metadata,
                  variable_access,
                  CapturedVariable,
                  span);
}

/// Creates debug information for a local variable introduced in the head of a match-statement arm.
///
/// Adds the created metadata nodes directly to the crate's IR.
pub fn create_match_binding_metadata(bcx: @mut Block,
                                     variable_ident: ast::Ident,
                                     node_id: ast::NodeId,
                                     variable_type: ty::t,
                                     span: Span) {
    if fn_should_be_ignored(bcx.fcx) {
        return;
    }

    let llptr = match bcx.fcx.lllocals.find_copy(&node_id) {
        Some(v) => v,
        None => {
            bcx.tcx().sess.span_bug(span, format!("No entry in lllocals table for {:?}", node_id));
        }
    };

    let scope_metadata = scope_metadata(bcx.fcx, node_id, span);

    declare_local(bcx,
                  variable_ident,
                  variable_type,
                  scope_metadata,
                  DirectVariable { alloca: llptr },
                  LocalVariable,
                  span);
}

/// Creates debug information for the self argument of a method.
///
/// Adds the created metadata nodes directly to the crate's IR.
pub fn create_self_argument_metadata(bcx: @mut Block,
                                     type_of_self: ty::t,
                                     llptr: ValueRef) {
    if fn_should_be_ignored(bcx.fcx) {
        return;
    }

    // Extract the span of the self argument from the method's AST
    let fnitem = bcx.ccx().tcx.items.get_copy(&bcx.fcx.id);
    let span = match fnitem {
        ast_map::node_method(@ast::method { explicit_self: explicit_self, _ }, _, _) => {
            explicit_self.span
        }
        ast_map::node_trait_method(
            @ast::provided(
                @ast::method {
                    explicit_self: explicit_self,
                    _
                }),
            _,
            _) => {
            explicit_self.span
        }
        _ => bcx.ccx().sess.bug(
                format!("create_self_argument_metadata: unexpected sort of node: {:?}", fnitem))
    };

    let scope_metadata = bcx.fcx.debug_context.get_ref(bcx.ccx(), span).fn_metadata;

    let argument_index = {
        let counter = &mut bcx.fcx.debug_context.get_mut_ref(bcx.ccx(), span).argument_counter;
        let argument_index = *counter;
        *counter += 1;
        argument_index
    };

    let address_operations = &[unsafe { llvm::LLVMDIBuilderCreateOpDeref(Type::i64().to_ref()) }];

    let variable_access = if unsafe { llvm::LLVMIsAAllocaInst(llptr) } != ptr::null() {
        DirectVariable { alloca: llptr }
    } else {
        // This is not stable and may break with future LLVM versions. llptr should really always
        // be an alloca. Anything else is not supported and just works by chance.
        IndirectVariable { alloca: llptr, address_operations: address_operations }
    };

    declare_local(bcx,
                  special_idents::self_,
                  type_of_self,
                  scope_metadata,
                  variable_access,
                  ArgumentVariable(argument_index),
                  span);
}

/// Creates debug information for the given function argument.
///
/// Adds the created metadata nodes directly to the crate's IR.
pub fn create_argument_metadata(bcx: @mut Block,
                                arg: &ast::arg) {
    if fn_should_be_ignored(bcx.fcx) {
        return;
    }

    let fcx = bcx.fcx;
    let cx = fcx.ccx;

    let def_map = cx.tcx.def_map;
    let scope_metadata = bcx.fcx.debug_context.get_ref(cx, arg.pat.span).fn_metadata;

    do pat_util::pat_bindings(def_map, arg.pat) |_, node_id, span, path_ref| {

        let llptr = match bcx.fcx.llargs.find_copy(&node_id) {
            Some(v) => v,
            None => {
                bcx.tcx().sess.span_bug(span,
                    format!("No entry in llargs table for {:?}", node_id));
            }
        };

        if unsafe { llvm::LLVMIsAAllocaInst(llptr) } == ptr::null() {
            cx.sess.span_bug(span, "debuginfo::create_argument_metadata() - \
                                    Referenced variable location is not an alloca!");
        }

        let argument_type = node_id_type(bcx, node_id);
        let argument_ident = ast_util::path_to_ident(path_ref);

        let argument_index = {
            let counter = &mut fcx.debug_context.get_mut_ref(cx, span).argument_counter;
            let argument_index = *counter;
            *counter += 1;
            argument_index
        };

        declare_local(bcx,
                      argument_ident,
                      argument_type,
                      scope_metadata,
                      DirectVariable { alloca: llptr },
                      ArgumentVariable(argument_index),
                      span);
    }
}

/// Sets the current debug location at the beginning of the span.
///
/// Maps to a call to llvm::LLVMSetCurrentDebugLocation(...). The node_id parameter is used to
/// reliably find the correct visibility scope for the code position.
pub fn set_source_location(fcx: &FunctionContext,
                           node_id: ast::NodeId,
                           span: Span) {
    if fn_should_be_ignored(fcx) {
        return;
    }

    let cx = fcx.ccx;

    debug!("set_source_location: {}", cx.sess.codemap.span_to_str(span));

    if fcx.debug_context.get_ref(cx, span).source_locations_enabled {
        let loc = span_start(cx, span);
        let scope = scope_metadata(fcx, node_id, span);

        set_debug_location(cx, DebugLocation::new(scope, loc.line, *loc.col));
    } else {
        set_debug_location(cx, UnknownLocation);
    }
}

/// Enables emitting source locations for the given functions.
///
/// Since we don't want source locations to be emitted for the function prelude, they are disabled
/// when beginning to translate a new function. This functions switches source location emitting on
/// and must therefore be called before the first real statement/expression of the function is
/// translated.
pub fn start_emitting_source_locations(fcx: &mut FunctionContext) {
    match fcx.debug_context {
        FunctionDebugContext(~ref mut data) => data.source_locations_enabled = true,
        _ => { /* safe to ignore */ }
    }
}

/// Creates the function-specific debug context.
///
/// Returns the FunctionDebugContext for the function which holds state needed for debug info
/// creation. The function may also return another variant of the FunctionDebugContext enum which
/// indicates why no debuginfo should be created for the function.
pub fn create_function_debug_context(cx: &mut CrateContext,
                                     fn_ast_id: ast::NodeId,
                                     param_substs: Option<@param_substs>,
                                     llfn: ValueRef) -> FunctionDebugContext {
    if !cx.sess.opts.debuginfo {
        return DebugInfoDisabled;
    }

    if fn_ast_id == -1 {
        return FunctionWithoutDebugInfo;
    }

    let empty_generics = ast::Generics { lifetimes: opt_vec::Empty, ty_params: opt_vec::Empty };

    let fnitem = cx.tcx.items.get_copy(&fn_ast_id);
    let (ident, fn_decl, generics, top_level_block, span, has_path) = match fnitem {
        ast_map::node_item(ref item, _) => {
            match item.node {
                ast::item_fn(ref fn_decl, _, _, ref generics, ref top_level_block) => {
                    (item.ident, fn_decl, generics, top_level_block, item.span, true)
                }
                _ => {
                    cx.sess.span_bug(item.span,
                        "create_function_debug_context: item bound to non-function");
                }
            }
        }
        ast_map::node_method(
            @ast::method {
                decl: ref fn_decl,
                ident: ident,
                generics: ref generics,
                body: ref top_level_block,
                span: span,
                _
            },
            _,
            _) => {
            (ident, fn_decl, generics, top_level_block, span, true)
        }
        ast_map::node_expr(ref expr) => {
            match expr.node {
                ast::ExprFnBlock(ref fn_decl, ref top_level_block) |
                ast::ExprProc(ref fn_decl, ref top_level_block) => {
                    let name = format!("fn{}", token::gensym("fn"));
                    let name = token::str_to_ident(name);
                    (name, fn_decl,
                        // This is not quite right. It should actually inherit the generics of the
                        // enclosing function.
                        &empty_generics,
                        top_level_block,
                        expr.span,
                        // Don't try to lookup the item path:
                        false)
                }
                _ => cx.sess.span_bug(expr.span,
                        "create_function_debug_context: expected an expr_fn_block here")
            }
        }
        ast_map::node_trait_method(
            @ast::provided(
                @ast::method {
                    decl: ref fn_decl,
                    ident: ident,
                    generics: ref generics,
                    body: ref top_level_block,
                    span: span,
                    _
                }),
            _,
            _) => {
            (ident, fn_decl, generics, top_level_block, span, true)
        }
        ast_map::node_foreign_item(@ast::foreign_item { _ }, _, _, _) |
        ast_map::node_variant(*) |
        ast_map::node_struct_ctor(*) => {
            return FunctionWithoutDebugInfo;
        }
        _ => cx.sess.bug(format!("create_function_debug_context: \
                                  unexpected sort of node: {:?}", fnitem))
    };

    // This can be the case for functions inlined from another crate
    if span == codemap::dummy_sp() {
        return FunctionWithoutDebugInfo;
    }

    let loc = span_start(cx, span);
    let file_metadata = file_metadata(cx, loc.file.name);

    let function_type_metadata = unsafe {
        let fn_signature = get_function_signature(cx, fn_ast_id, fn_decl, param_substs, span);
        llvm::LLVMDIBuilderCreateSubroutineType(DIB(cx), file_metadata, fn_signature)
    };

    // get_template_parameters() will append a `<...>` clause to the function name if necessary.
    let mut function_name = token::ident_to_str(&ident).to_owned();
    let template_parameters = get_template_parameters(cx,
                                                      generics,
                                                      param_substs,
                                                      file_metadata,
                                                      &mut function_name);

    // There is no ast_map::path for ast::ExprFnBlock-type functions. For now, just don't put them
    // into a namespace. In the future this could be improved somehow (storing a path in the
    // ast_map, or construct a path using the enclosing function).
    let (linkage_name, containing_scope) = if has_path {
        let namespace_node = namespace_for_item(cx, ast_util::local_def(fn_ast_id), span);
        let linkage_name = namespace_node.mangled_name_of_contained_item(function_name);
        let containing_scope = namespace_node.scope;
        (linkage_name, containing_scope)
    } else {
        (function_name.clone(), file_metadata)
    };

    let scope_line = get_scope_line(cx, top_level_block, loc.line);

    let fn_metadata = do function_name.with_c_str |function_name| {
                      do linkage_name.with_c_str |linkage_name| {
        unsafe {
            llvm::LLVMDIBuilderCreateFunction(
                DIB(cx),
                containing_scope,
                function_name,
                linkage_name,
                file_metadata,
                loc.line as c_uint,
                function_type_metadata,
                false,
                true,
                scope_line as c_uint,
                FlagPrototyped as c_uint,
                cx.sess.opts.optimize != session::No,
                llfn,
                template_parameters,
                ptr::null())
        }
    }};

    // Initialize fn debug context (including scope map and namespace map)
    let mut fn_debug_context = ~FunctionDebugContextData {
        scope_map: HashMap::new(),
        fn_metadata: fn_metadata,
        argument_counter: 1,
        source_locations_enabled: false,
    };

    let arg_pats = do fn_decl.inputs.map |arg_ref| { arg_ref.pat };
    populate_scope_map(cx, arg_pats, top_level_block, fn_metadata, &mut fn_debug_context.scope_map);

    return FunctionDebugContext(fn_debug_context);

    fn get_function_signature(cx: &mut CrateContext,
                              fn_ast_id: ast::NodeId,
                              fn_decl: &ast::fn_decl,
                              param_substs: Option<@param_substs>,
                              error_span: Span) -> DIArray {
        if !cx.sess.opts.extra_debuginfo {
            return create_DIArray(DIB(cx), []);
        }

        let mut signature = vec::with_capacity(fn_decl.inputs.len() + 1);

        // Return type -- llvm::DIBuilder wants this at index 0
        match fn_decl.output.node {
            ast::ty_nil => {
                signature.push(ptr::null());
            }
            _ => {
                assert_type_for_node_id(cx, fn_ast_id, error_span);

                let return_type = ty::node_id_to_type(cx.tcx, fn_ast_id);
                let return_type = match param_substs {
                    None => return_type,
                    Some(substs) => {
                        ty::subst_tps(cx.tcx, substs.tys, substs.self_ty, return_type)
                    }
                };

                signature.push(type_metadata(cx, return_type, codemap::dummy_sp()));
            }
        }

        // Arguments types
        for arg in fn_decl.inputs.iter() {
            assert_type_for_node_id(cx, arg.pat.id, arg.pat.span);
            let arg_type = ty::node_id_to_type(cx.tcx, arg.pat.id);
            let arg_type = match param_substs {
                None => arg_type,
                Some(substs) => {
                    ty::subst_tps(cx.tcx, substs.tys, substs.self_ty, arg_type)
                }
            };

            signature.push(type_metadata(cx, arg_type, codemap::dummy_sp()));
        }

        return create_DIArray(DIB(cx), signature);
    }

    fn get_template_parameters(cx: &mut CrateContext,
                               generics: &ast::Generics,
                               param_substs: Option<@param_substs>,
                               file_metadata: DIFile,
                               name_to_append_suffix_to: &mut ~str)
                            -> DIArray {
        let self_type = match param_substs {
            Some(@param_substs{ self_ty: self_type, _ }) => self_type,
            _ => None
        };

        // Only true for static default methods:
        let has_self_type = self_type.is_some();

        if !generics.is_type_parameterized() && !has_self_type {
            return create_DIArray(DIB(cx), []);
        }

        name_to_append_suffix_to.push_char('<');

        // The list to be filled with template parameters:
        let mut template_params: ~[DIDescriptor] = vec::with_capacity(generics.ty_params.len() + 1);

        // Handle self type
        if has_self_type {
            let actual_self_type = self_type.unwrap();
            // Add self type name to <...> clause of function name
            let actual_self_type_name = ppaux::ty_to_str(cx.tcx, actual_self_type);
            name_to_append_suffix_to.push_str(actual_self_type_name);

            if generics.is_type_parameterized() {
                name_to_append_suffix_to.push_str(",");
            }

            // Only create type information if extra_debuginfo is enabled
            if cx.sess.opts.extra_debuginfo {
                let actual_self_type_metadata = type_metadata(cx,
                                                              actual_self_type,
                                                              codemap::dummy_sp());

                let ident = special_idents::type_self;

                let param_metadata = do token::ident_to_str(&ident).with_c_str |name| {
                    unsafe {
                        llvm::LLVMDIBuilderCreateTemplateTypeParameter(
                            DIB(cx),
                            file_metadata,
                            name,
                            actual_self_type_metadata,
                            ptr::null(),
                            0,
                            0)
                    }
                };

                template_params.push(param_metadata);
            }
        }

        // Handle other generic parameters
        let actual_types = match param_substs {
            Some(@param_substs { tys: ref types, _ }) => types,
            None => {
                return create_DIArray(DIB(cx), template_params);
            }
        };

        for (index, &ast::TyParam{ ident: ident, _ }) in generics.ty_params.iter().enumerate() {
            let actual_type = actual_types[index];
            // Add actual type name to <...> clause of function name
            let actual_type_name = ppaux::ty_to_str(cx.tcx, actual_type);
            name_to_append_suffix_to.push_str(actual_type_name);

            if index != generics.ty_params.len() - 1 {
                name_to_append_suffix_to.push_str(",");
            }

            // Again, only create type information if extra_debuginfo is enabled
            if cx.sess.opts.extra_debuginfo {
                let actual_type_metadata = type_metadata(cx, actual_type, codemap::dummy_sp());
                let param_metadata = do token::ident_to_str(&ident).with_c_str |name| {
                    unsafe {
                        llvm::LLVMDIBuilderCreateTemplateTypeParameter(
                            DIB(cx),
                            file_metadata,
                            name,
                            actual_type_metadata,
                            ptr::null(),
                            0,
                            0)
                    }
                };
                template_params.push(param_metadata);
            }
        }

        name_to_append_suffix_to.push_char('>');

        return create_DIArray(DIB(cx), template_params);
    }

    fn get_scope_line(cx: &CrateContext,
                      top_level_block: &ast::Block,
                      default: uint)
                   -> uint {
        match *top_level_block {
            ast::Block { stmts: ref statements, _ } if statements.len() > 0 => {
                span_start(cx, statements[0].span).line
            }
            ast::Block { expr: Some(@ref expr), _ } => {
                span_start(cx, expr.span).line
            }
            _ => default
        }
    }
}

//=-------------------------------------------------------------------------------------------------
// Module-Internal debug info creation functions
//=-------------------------------------------------------------------------------------------------

fn create_DIArray(builder: DIBuilderRef, arr: &[DIDescriptor]) -> DIArray {
    return unsafe {
        llvm::LLVMDIBuilderGetOrCreateArray(builder, vec::raw::to_ptr(arr), arr.len() as u32)
    };
}

fn compile_unit_metadata(cx: @mut CrateContext) {
    let dcx = debug_context(cx);
    let crate_name: &str = dcx.crate_file;

    debug!("compile_unit_metadata: {:?}", crate_name);

    // FIXME (#9639): This needs to handle non-utf8 paths
    let work_dir = cx.sess.working_dir.as_str().unwrap();
    let producer = format!("rustc version {}", env!("CFG_VERSION"));

    do crate_name.with_c_str |crate_name| {
    do work_dir.with_c_str |work_dir| {
    do producer.with_c_str |producer| {
    do "".with_c_str |flags| {
    do "".with_c_str |split_name| {
        unsafe {
            llvm::LLVMDIBuilderCreateCompileUnit(
                dcx.builder,
                DW_LANG_RUST,
                crate_name,
                work_dir,
                producer,
                cx.sess.opts.optimize != session::No,
                flags,
                0,
                split_name);
        }
    }}}}};
}

fn declare_local(bcx: @mut Block,
                 variable_ident: ast::Ident,
                 variable_type: ty::t,
                 scope_metadata: DIScope,
                 variable_access: VariableAccess,
                 variable_kind: VariableKind,
                 span: Span) {
    let cx: &mut CrateContext = bcx.ccx();

    let filename = span_start(cx, span).file.name;
    let file_metadata = file_metadata(cx, filename);

    let name: &str = token::ident_to_str(&variable_ident);
    let loc = span_start(cx, span);
    let type_metadata = type_metadata(cx, variable_type, span);

    let argument_index = match variable_kind {
        ArgumentVariable(index) => index,
        LocalVariable    |
        CapturedVariable => 0
    } as c_uint;

    let (var_alloca, var_metadata) = do name.with_c_str |name| {
        match variable_access {
            DirectVariable { alloca } => (
                alloca,
                unsafe {
                    llvm::LLVMDIBuilderCreateLocalVariable(
                        DIB(cx),
                        DW_TAG_auto_variable,
                        scope_metadata,
                        name,
                        file_metadata,
                        loc.line as c_uint,
                        type_metadata,
                        cx.sess.opts.optimize != session::No,
                        0,
                        argument_index)
                }
            ),
            IndirectVariable { alloca, address_operations } => (
                alloca,
                unsafe {
                    llvm::LLVMDIBuilderCreateComplexVariable(
                        DIB(cx),
                        DW_TAG_auto_variable,
                        scope_metadata,
                        name,
                        file_metadata,
                        loc.line as c_uint,
                        type_metadata,
                        vec::raw::to_ptr(address_operations),
                        address_operations.len() as c_uint,
                        argument_index)
                }
            )
        }
    };

    set_debug_location(cx, DebugLocation::new(scope_metadata, loc.line, *loc.col));
    unsafe {
        let instr = llvm::LLVMDIBuilderInsertDeclareAtEnd(
            DIB(cx),
            var_alloca,
            var_metadata,
            bcx.llbb);

        llvm::LLVMSetInstDebugLocation(trans::build::B(bcx).llbuilder, instr);
    }

    match variable_kind {
        ArgumentVariable(_) | CapturedVariable => {
            assert!(!bcx.fcx.debug_context.get_ref(cx, span).source_locations_enabled);
            set_debug_location(cx, UnknownLocation);
        }
        _ => { /* nothing to do */ }
    }
}

fn file_metadata(cx: &mut CrateContext, full_path: &str) -> DIFile {
    match debug_context(cx).created_files.find_equiv(&full_path) {
        Some(file_metadata) => return *file_metadata,
        None => ()
    }

    debug!("file_metadata: {}", full_path);

    // FIXME (#9639): This needs to handle non-utf8 paths
    let work_dir = cx.sess.working_dir.as_str().unwrap();
    let file_name =
        if full_path.starts_with(work_dir) {
            full_path.slice(work_dir.len() + 1u, full_path.len())
        } else {
            full_path
        };

    let file_metadata =
        do file_name.with_c_str |file_name| {
        do work_dir.with_c_str |work_dir| {
            unsafe {
                llvm::LLVMDIBuilderCreateFile(DIB(cx), file_name, work_dir)
            }
        }};

    debug_context(cx).created_files.insert(full_path.to_owned(), file_metadata);
    return file_metadata;
}

/// Finds the scope metadata node for the given AST node.
fn scope_metadata(fcx: &FunctionContext,
                  node_id: ast::NodeId,
                  span: Span)
               -> DIScope {
    let scope_map = &fcx.debug_context.get_ref(fcx.ccx, span).scope_map;

    match scope_map.find_copy(&node_id) {
        Some(scope_metadata) => scope_metadata,
        None => {
            let node = fcx.ccx.tcx.items.get_copy(&node_id);

            fcx.ccx.sess.span_bug(span,
                format!("debuginfo: Could not find scope info for node {:?}", node));
        }
    }
}

fn basic_type_metadata(cx: &mut CrateContext, t: ty::t) -> DIType {

    debug!("basic_type_metadata: {:?}", ty::get(t));

    let (name, encoding) = match ty::get(t).sty {
        ty::ty_nil | ty::ty_bot => (~"uint", DW_ATE_unsigned),
        ty::ty_bool => (~"bool", DW_ATE_boolean),
        ty::ty_char => (~"char", DW_ATE_unsigned_char),
        ty::ty_int(int_ty) => match int_ty {
            ast::ty_i => (~"int", DW_ATE_signed),
            ast::ty_i8 => (~"i8", DW_ATE_signed),
            ast::ty_i16 => (~"i16", DW_ATE_signed),
            ast::ty_i32 => (~"i32", DW_ATE_signed),
            ast::ty_i64 => (~"i64", DW_ATE_signed)
        },
        ty::ty_uint(uint_ty) => match uint_ty {
            ast::ty_u => (~"uint", DW_ATE_unsigned),
            ast::ty_u8 => (~"u8", DW_ATE_unsigned),
            ast::ty_u16 => (~"u16", DW_ATE_unsigned),
            ast::ty_u32 => (~"u32", DW_ATE_unsigned),
            ast::ty_u64 => (~"u64", DW_ATE_unsigned)
        },
        ty::ty_float(float_ty) => match float_ty {
            ast::ty_f32 => (~"f32", DW_ATE_float),
            ast::ty_f64 => (~"f64", DW_ATE_float)
        },
        _ => cx.sess.bug("debuginfo::basic_type_metadata - t is invalid type")
    };

    let llvm_type = type_of::type_of(cx, t);
    let (size, align) = size_and_align_of(cx, llvm_type);
    let ty_metadata = do name.with_c_str |name| {
        unsafe {
            llvm::LLVMDIBuilderCreateBasicType(
                DIB(cx),
                name,
                bytes_to_bits(size),
                bytes_to_bits(align),
                encoding)
        }
    };

    return ty_metadata;
}

fn pointer_type_metadata(cx: &mut CrateContext,
                         pointer_type: ty::t,
                         pointee_type_metadata: DIType)
                      -> DIType {
    let pointer_llvm_type = type_of::type_of(cx, pointer_type);
    let (pointer_size, pointer_align) = size_and_align_of(cx, pointer_llvm_type);
    let name = ppaux::ty_to_str(cx.tcx, pointer_type);
    let ptr_metadata = do name.with_c_str |name| {
        unsafe {
            llvm::LLVMDIBuilderCreatePointerType(
                DIB(cx),
                pointee_type_metadata,
                bytes_to_bits(pointer_size),
                bytes_to_bits(pointer_align),
                name)
        }
    };
    return ptr_metadata;
}

trait MemberDescriptionFactory {
    fn create_member_descriptions(&self, cx: &mut CrateContext)
                                  -> ~[MemberDescription];
}

struct StructMemberDescriptionFactory {
    fields: ~[ty::field],
    span: Span,
}

impl MemberDescriptionFactory for StructMemberDescriptionFactory {
    fn create_member_descriptions(&self, cx: &mut CrateContext)
                                  -> ~[MemberDescription] {
        do self.fields.map |field| {
            let name = if field.ident.name == special_idents::unnamed_field.name {
                @""
            } else {
                token::ident_to_str(&field.ident)
            };

            MemberDescription {
                name: name,
                llvm_type: type_of::type_of(cx, field.mt.ty),
                type_metadata: type_metadata(cx, field.mt.ty, self.span),
                offset: ComputedMemberOffset,
            }
        }
    }
}

fn prepare_struct_metadata(cx: &mut CrateContext,
                           struct_type: ty::t,
                           def_id: ast::DefId,
                           substs: &ty::substs,
                           span: Span)
                        -> RecursiveTypeDescription {
    let struct_name = ppaux::ty_to_str(cx.tcx, struct_type);
    let struct_llvm_type = type_of::type_of(cx, struct_type);

    let (containing_scope, definition_span) = get_namespace_and_span_for_item(cx, def_id, span);

    let file_name = span_start(cx, definition_span).file.name;
    let file_metadata = file_metadata(cx, file_name);

    let struct_metadata_stub = create_struct_stub(cx,
                                                  struct_llvm_type,
                                                  struct_name,
                                                  containing_scope,
                                                  file_metadata,
                                                  definition_span);

    let fields = ty::struct_fields(cx.tcx, def_id, substs);

    UnfinishedMetadata {
        cache_id: cache_id_for_type(struct_type),
        metadata_stub: struct_metadata_stub,
        llvm_type: struct_llvm_type,
        file_metadata: file_metadata,
        member_description_factory: @StructMemberDescriptionFactory {
            fields: fields,
            span: span,
        } as @MemberDescriptionFactory,
    }
}

enum RecursiveTypeDescription {
    UnfinishedMetadata {
        cache_id: uint,
        metadata_stub: DICompositeType,
        llvm_type: Type,
        file_metadata: DIFile,
        member_description_factory: @MemberDescriptionFactory,
    },
    FinalMetadata(DICompositeType)
}

impl RecursiveTypeDescription {

    fn metadata(&self) -> DICompositeType {
        match *self {
            UnfinishedMetadata { metadata_stub, _ } => metadata_stub,
            FinalMetadata(metadata) => metadata
        }
    }

    fn finalize(&self, cx: &mut CrateContext) -> DICompositeType {
        match *self {
            FinalMetadata(metadata) => metadata,
            UnfinishedMetadata {
                cache_id,
                metadata_stub,
                llvm_type,
                file_metadata,
                member_description_factory
            } => {
                // Insert the stub into the cache in order to allow recursive references ...
                debug_context(cx).created_types.insert(cache_id, metadata_stub);

                // ... then create the member descriptions ...
                let member_descriptions = member_description_factory.
                    create_member_descriptions(cx);

                // ... and attach them to the stub to complete it.
                set_members_of_composite_type(cx,
                                              metadata_stub,
                                              llvm_type,
                                              member_descriptions,
                                              file_metadata,
                                              codemap::dummy_sp());
                return metadata_stub;
            }
        }
    }
}

struct TupleMemberDescriptionFactory {
    component_types: ~[ty::t],
    span: Span,
}

impl MemberDescriptionFactory for TupleMemberDescriptionFactory {
    fn create_member_descriptions(&self, cx: &mut CrateContext)
                                  -> ~[MemberDescription] {
        do self.component_types.map |&component_type| {
            MemberDescription {
                name: @"",
                llvm_type: type_of::type_of(cx, component_type),
                type_metadata: type_metadata(cx, component_type, self.span),
                offset: ComputedMemberOffset,
            }
        }
    }
}

fn prepare_tuple_metadata(cx: &mut CrateContext,
                          tuple_type: ty::t,
                          component_types: &[ty::t],
                          span: Span)
                       -> RecursiveTypeDescription {
    let tuple_name = ppaux::ty_to_str(cx.tcx, tuple_type);
    let tuple_llvm_type = type_of::type_of(cx, tuple_type);

    let loc = span_start(cx, span);
    let file_metadata = file_metadata(cx, loc.file.name);

    UnfinishedMetadata {
        cache_id: cache_id_for_type(tuple_type),
        metadata_stub: create_struct_stub(cx,
                                          tuple_llvm_type,
                                          tuple_name,
                                          file_metadata,
                                          file_metadata,
                                          span),
        llvm_type: tuple_llvm_type,
        file_metadata: file_metadata,
        member_description_factory: @TupleMemberDescriptionFactory {
            component_types: component_types.to_owned(),
            span: span,
        } as @MemberDescriptionFactory
    }
}

struct GeneralMemberDescriptionFactory {
    type_rep: @adt::Repr,
    variants: @~[@ty::VariantInfo],
    discriminant_type_metadata: ValueRef,
    containing_scope: DIScope,
    file_metadata: DIFile,
    span: Span,
}

impl MemberDescriptionFactory for GeneralMemberDescriptionFactory {
    fn create_member_descriptions(&self, cx: &mut CrateContext)
                                  -> ~[MemberDescription] {
        // Capture type_rep, so we don't have to copy the struct_defs array
        let struct_defs = match *self.type_rep {
            adt::General(ref struct_defs) => struct_defs,
            _ => cx.sess.bug("unreachable")
        };

        do struct_defs
            .iter()
            .enumerate()
            .map |(i, struct_def)| {
                let (variant_type_metadata, variant_llvm_type, member_desc_factory) =
                    describe_variant(cx,
                                     struct_def,
                                     self.variants[i],
                                     Some(self.discriminant_type_metadata),
                                     self.containing_scope,
                                     self.file_metadata,
                                     self.span);

                let member_descriptions =
                    member_desc_factory.create_member_descriptions(cx);

                set_members_of_composite_type(cx,
                                              variant_type_metadata,
                                              variant_llvm_type,
                                              member_descriptions,
                                              self.file_metadata,
                                              codemap::dummy_sp());
                MemberDescription {
                    name: @"",
                    llvm_type: variant_llvm_type,
                    type_metadata: variant_type_metadata,
                    offset: FixedMemberOffset { bytes: 0 },
                }
        }.collect()
    }
}

struct EnumVariantMemberDescriptionFactory {
    args: ~[(@str, ty::t)],
    discriminant_type_metadata: Option<DIType>,
    span: Span,
}

impl MemberDescriptionFactory for EnumVariantMemberDescriptionFactory {
    fn create_member_descriptions(&self, cx: &mut CrateContext)
                                  -> ~[MemberDescription] {
        do self.args.iter().enumerate().map |(i, &(name, ty))| {
            MemberDescription {
                name: name,
                llvm_type: type_of::type_of(cx, ty),
                type_metadata: match self.discriminant_type_metadata {
                    Some(metadata) if i == 0 => metadata,
                    _ => type_metadata(cx, ty, self.span)
                },
                offset: ComputedMemberOffset,
            }
        }.collect()
    }
}

fn describe_variant(cx: &mut CrateContext,
                    struct_def: &adt::Struct,
                    variant_info: &ty::VariantInfo,
                    discriminant_type_metadata: Option<DIType>,
                    containing_scope: DIScope,
                    file_metadata: DIFile,
                    span: Span)
                 -> (DICompositeType, Type, @MemberDescriptionFactory) {
    let variant_name = token::ident_to_str(&variant_info.name);
    let variant_llvm_type = Type::struct_(struct_def.fields.map(|&t| type_of::type_of(cx, t)),
                                          struct_def.packed);
    // Could some consistency checks here: size, align, field count, discr type

    // Find the source code location of the variant's definition
    let variant_definition_span = if variant_info.id.crate == ast::LOCAL_CRATE {
        match cx.tcx.items.find(&variant_info.id.node) {
            Some(&ast_map::node_variant(ref variant, _, _)) => variant.span,
            ref node => {
                cx.sess.span_warn(span,
                    format!("debuginfo::enum_metadata()::adt_struct_metadata() - Unexpected node \
                          type: {:?}. This is a bug.", node));
                codemap::dummy_sp()
            }
        }
    } else {
        // For definitions from other crates we have no location information available.
        codemap::dummy_sp()
    };

    let metadata_stub = create_struct_stub(cx,
                                           variant_llvm_type,
                                           variant_name,
                                           containing_scope,
                                           file_metadata,
                                           variant_definition_span);

    // Get the argument names from the enum variant info
    let mut arg_names = match variant_info.arg_names {
        Some(ref names) => do names.map |ident| { token::ident_to_str(ident) },
        None => do variant_info.args.map |_| { @"" }
    };

    // If this is not a univariant enum, there is also the (unnamed) discriminant field
    if discriminant_type_metadata.is_some() {
        arg_names.insert(0, @"");
    }

    // Build an array of (field name, field type) pairs to be captured in the factory closure.
    let args: ~[(@str, ty::t)] = arg_names.iter()
        .zip(struct_def.fields.iter())
        .map(|(&s, &t)| (s, t))
        .collect();

    let member_description_factory =
        @EnumVariantMemberDescriptionFactory {
            args: args,
            discriminant_type_metadata: discriminant_type_metadata,
            span: span,
        } as @MemberDescriptionFactory;

    (metadata_stub, variant_llvm_type, member_description_factory)
}

fn prepare_enum_metadata(cx: &mut CrateContext,
                         enum_type: ty::t,
                         enum_def_id: ast::DefId,
                         span: Span)
                      -> RecursiveTypeDescription {
    let enum_name = ppaux::ty_to_str(cx.tcx, enum_type);

    let (containing_scope, definition_span) = get_namespace_and_span_for_item(cx,
                                                                              enum_def_id,
                                                                              span);
    let loc = span_start(cx, definition_span);
    let file_metadata = file_metadata(cx, loc.file.name);

    // For empty enums there is an early exit. Just describe it as an empty struct with the
    // appropriate type name
    if ty::type_is_empty(cx.tcx, enum_type) {
        let empty_type_metadata = composite_type_metadata(cx,
                                                          Type::nil(),
                                                          enum_name,
                                                          [],
                                                          containing_scope,
                                                          file_metadata,
                                                          definition_span);

        return FinalMetadata(empty_type_metadata);
    }

    // Prepare some data (llvm type, size, align, etc) about the discriminant. This data will be
    // needed in all of the following cases.
    let discriminant_llvm_type = Type::enum_discrim(cx);
    let (discriminant_size, discriminant_align) = size_and_align_of(cx, discriminant_llvm_type);

    assert!(Type::enum_discrim(cx) == cx.int_type);
    let discriminant_base_type_metadata = type_metadata(cx, ty::mk_int(), codemap::dummy_sp());

    let variants = ty::enum_variants(cx.tcx, enum_def_id);

    let enumerators_metadata: ~[DIDescriptor] = variants
        .iter()
        .map(|v| {
            let name: &str = token::ident_to_str(&v.name);
            let discriminant_value = v.disr_val as c_ulonglong;

            do name.with_c_str |name| {
                unsafe {
                    llvm::LLVMDIBuilderCreateEnumerator(
                        DIB(cx),
                        name,
                        discriminant_value)
                }
            }
        })
        .collect();

    let discriminant_type_metadata = do enum_name.with_c_str |enum_name| {
        unsafe {
            llvm::LLVMDIBuilderCreateEnumerationType(
                DIB(cx),
                containing_scope,
                enum_name,
                file_metadata,
                loc.line as c_uint,
                bytes_to_bits(discriminant_size),
                bytes_to_bits(discriminant_align),
                create_DIArray(DIB(cx), enumerators_metadata),
                discriminant_base_type_metadata)
        }
    };

    let type_rep = adt::represent_type(cx, enum_type);

    return match *type_rep {
        adt::CEnum(*) => {
            FinalMetadata(discriminant_type_metadata)
        }
        adt::Univariant(ref struct_def, _) => {
            assert!(variants.len() == 1);
            let (metadata_stub,
                 variant_llvm_type,
                 member_description_factory) = describe_variant(cx,
                                                                struct_def,
                                                                variants[0],
                                                                None,
                                                                containing_scope,
                                                                file_metadata,
                                                                span);
            UnfinishedMetadata {
                cache_id: cache_id_for_type(enum_type),
                metadata_stub: metadata_stub,
                llvm_type: variant_llvm_type,
                file_metadata: file_metadata,
                member_description_factory: member_description_factory
            }
        }
        adt::General(_) => {
            let enum_llvm_type = type_of::type_of(cx, enum_type);
            let (enum_type_size, enum_type_align) = size_and_align_of(cx, enum_llvm_type);

            let enum_metadata = do enum_name.with_c_str |enum_name| {
                unsafe {
                    llvm::LLVMDIBuilderCreateUnionType(
                    DIB(cx),
                    containing_scope,
                    enum_name,
                    file_metadata,
                    loc.line as c_uint,
                    bytes_to_bits(enum_type_size),
                    bytes_to_bits(enum_type_align),
                    0, // Flags
                    ptr::null(),
                    0) // RuntimeLang
            }};

            UnfinishedMetadata {
                cache_id: cache_id_for_type(enum_type),
                metadata_stub: enum_metadata,
                llvm_type: enum_llvm_type,
                file_metadata: file_metadata,
                member_description_factory: @GeneralMemberDescriptionFactory {
                    type_rep: type_rep,
                    variants: variants,
                    discriminant_type_metadata: discriminant_type_metadata,
                    containing_scope: containing_scope,
                    file_metadata: file_metadata,
                    span: span,
                } as @MemberDescriptionFactory,
            }
        }
        adt::NullablePointer { nonnull: ref struct_def, nndiscr, _ } => {
            let (metadata_stub,
                 variant_llvm_type,
                 member_description_factory) = describe_variant(cx,
                                                                struct_def,
                                                                variants[nndiscr],
                                                                None,
                                                                containing_scope,
                                                                file_metadata,
                                                                span);
            UnfinishedMetadata {
                cache_id: cache_id_for_type(enum_type),
                metadata_stub: metadata_stub,
                llvm_type: variant_llvm_type,
                file_metadata: file_metadata,
                member_description_factory: member_description_factory
            }
        }
    };
}

enum MemberOffset {
    FixedMemberOffset { bytes: uint },
    // For ComputedMemberOffset, the offset is read from the llvm type definition
    ComputedMemberOffset
}

struct MemberDescription {
    name: @str,
    llvm_type: Type,
    type_metadata: DIType,
    offset: MemberOffset,
}

/// Creates debug information for a composite type, that is, anything that results in a LLVM struct.
///
/// Examples of Rust types to use this are: structs, tuples, boxes, vecs, and enums.
fn composite_type_metadata(cx: &mut CrateContext,
                           composite_llvm_type: Type,
                           composite_type_name: &str,
                           member_descriptions: &[MemberDescription],
                           containing_scope: DIScope,
                           file_metadata: DIFile,
                           definition_span: Span)
                        -> DICompositeType {
    // Create the (empty) struct metadata node ...
    let composite_type_metadata = create_struct_stub(cx,
                                                     composite_llvm_type,
                                                     composite_type_name,
                                                     containing_scope,
                                                     file_metadata,
                                                     definition_span);

    // ... and immediately create and add the member descriptions.
    set_members_of_composite_type(cx,
                                  composite_type_metadata,
                                  composite_llvm_type,
                                  member_descriptions,
                                  file_metadata,
                                  definition_span);

    return composite_type_metadata;
}

fn set_members_of_composite_type(cx: &mut CrateContext,
                                 composite_type_metadata: DICompositeType,
                                 composite_llvm_type: Type,
                                 member_descriptions: &[MemberDescription],
                                 file_metadata: DIFile,
                                 definition_span: Span) {
    // In some rare cases LLVM metadata uniquing would lead to an existing type description being
    // used instead of a new one created in create_struct_stub. This would cause a hard to trace
    // assertion in DICompositeType::SetTypeArray(). The following check makes sure that we get a
    // better error message if this should happen again due to some regression.
    if debug_context(cx).composite_types_completed.contains(&composite_type_metadata) {
        cx.sess.span_bug(definition_span, "debuginfo::set_members_of_composite_type() - Already \
                                           completed forward declaration re-encountered.");
    } else {
        debug_context(cx).composite_types_completed.insert(composite_type_metadata);
    }

    let loc = span_start(cx, definition_span);

    let member_metadata: ~[DIDescriptor] = member_descriptions
        .iter()
        .enumerate()
        .map(|(i, member_description)| {
            let (member_size, member_align) = size_and_align_of(cx, member_description.llvm_type);
            let member_offset = match member_description.offset {
                FixedMemberOffset { bytes } => bytes,
                ComputedMemberOffset => machine::llelement_offset(cx, composite_llvm_type, i)
            };

            do member_description.name.with_c_str |member_name| {
                unsafe {
                    llvm::LLVMDIBuilderCreateMemberType(
                        DIB(cx),
                        composite_type_metadata,
                        member_name,
                        file_metadata,
                        loc.line as c_uint,
                        bytes_to_bits(member_size),
                        bytes_to_bits(member_align),
                        bytes_to_bits(member_offset),
                        0,
                        member_description.type_metadata)
                }
            }
        })
        .collect();

    unsafe {
        let type_array = create_DIArray(DIB(cx), member_metadata);
        llvm::LLVMDICompositeTypeSetTypeArray(composite_type_metadata, type_array);
    }
}

// A convenience wrapper around LLVMDIBuilderCreateStructType(). Does not do any caching, does not
// add any fields to the struct. This can be done later with set_members_of_composite_type().
fn create_struct_stub(cx: &mut CrateContext,
                      struct_llvm_type: Type,
                      struct_type_name: &str,
                      containing_scope: DIScope,
                      file_metadata: DIFile,
                      definition_span: Span)
                   -> DICompositeType {
    let loc = span_start(cx, definition_span);
    let (struct_size, struct_align) = size_and_align_of(cx, struct_llvm_type);

    // We assign unique IDs to the type stubs so LLVM metadata uniquing does not reuse instances
    // where we don't want it.
    let unique_id = unsafe {
        static mut unique_id_counter: atomics::AtomicUint = atomics::INIT_ATOMIC_UINT;
        format!("DiStructStub{}", unique_id_counter.fetch_add(1, atomics::SeqCst))
    };

    return unsafe {
        do struct_type_name.with_c_str |name| {
        do unique_id.with_c_str |unique_id| {
            // LLVMDIBuilderCreateStructType() wants an empty array. A null pointer will lead to
            // hard to trace and debug LLVM assertions later on in llvm/lib/IR/Value.cpp
            let empty_array = create_DIArray(DIB(cx), []);

            llvm::LLVMDIBuilderCreateStructType(
                DIB(cx),
                containing_scope,
                name,
                file_metadata,
                loc.line as c_uint,
                bytes_to_bits(struct_size),
                bytes_to_bits(struct_align),
                0,
                ptr::null(),
                empty_array,
                0,
                ptr::null(),
                unique_id)
    }}};
}

fn boxed_type_metadata(cx: &mut CrateContext,
                       content_type_name: Option<&str>,
                       content_llvm_type: Type,
                       content_type_metadata: DIType,
                       span: Span)
                    -> DICompositeType {
    let box_type_name = match content_type_name {
        Some(content_type_name) => format!("Boxed<{}>", content_type_name),
        None                    => ~"BoxedType"
    };

    let box_llvm_type = Type::box(cx, &content_llvm_type);
    let member_llvm_types = box_llvm_type.field_types();
    assert!(box_layout_is_correct(cx, member_llvm_types, content_llvm_type));

    let int_type = ty::mk_int();
    let nil_pointer_type = ty::mk_nil_ptr(cx.tcx);
    let nil_pointer_type_metadata = type_metadata(cx, nil_pointer_type, codemap::dummy_sp());

    let member_descriptions = [
        MemberDescription {
            name: @"refcnt",
            llvm_type: member_llvm_types[0],
            type_metadata: type_metadata(cx, int_type, codemap::dummy_sp()),
            offset: ComputedMemberOffset,
        },
        MemberDescription {
            name: @"tydesc",
            llvm_type: member_llvm_types[1],
            type_metadata: nil_pointer_type_metadata,
            offset: ComputedMemberOffset,
        },
        MemberDescription {
            name: @"prev",
            llvm_type: member_llvm_types[2],
            type_metadata: nil_pointer_type_metadata,
            offset: ComputedMemberOffset,
        },
        MemberDescription {
            name: @"next",
            llvm_type: member_llvm_types[3],
            type_metadata: nil_pointer_type_metadata,
            offset: ComputedMemberOffset,
        },
        MemberDescription {
            name: @"val",
            llvm_type: member_llvm_types[4],
            type_metadata: content_type_metadata,
            offset: ComputedMemberOffset,
        }
    ];

    let loc = span_start(cx, span);
    let file_metadata = file_metadata(cx, loc.file.name);

    return composite_type_metadata(
        cx,
        box_llvm_type,
        box_type_name,
        member_descriptions,
        file_metadata,
        file_metadata,
        span);

    // Unfortunately, we cannot assert anything but the correct types here---and not whether the
    // 'next' and 'prev' pointers are in the correct order.
    fn box_layout_is_correct(cx: &CrateContext,
                             member_llvm_types: &[Type],
                             content_llvm_type: Type)
                          -> bool {
        member_llvm_types.len() == 5 &&
        member_llvm_types[0] == cx.int_type &&
        member_llvm_types[1] == cx.tydesc_type.ptr_to() &&
        member_llvm_types[2] == Type::i8().ptr_to() &&
        member_llvm_types[3] == Type::i8().ptr_to() &&
        member_llvm_types[4] == content_llvm_type
    }
}

fn fixed_vec_metadata(cx: &mut CrateContext,
                      element_type: ty::t,
                      len: uint,
                      span: Span)
                   -> DIType {
    let element_type_metadata = type_metadata(cx, element_type, span);
    let element_llvm_type = type_of::type_of(cx, element_type);
    let (element_type_size, element_type_align) = size_and_align_of(cx, element_llvm_type);

    let subrange = unsafe {
        llvm::LLVMDIBuilderGetOrCreateSubrange(
        DIB(cx),
        0,
        len as c_longlong)
    };

    let subscripts = create_DIArray(DIB(cx), [subrange]);
    return unsafe {
        llvm::LLVMDIBuilderCreateArrayType(
            DIB(cx),
            bytes_to_bits(element_type_size * len),
            bytes_to_bits(element_type_align),
            element_type_metadata,
            subscripts)
    };
}

fn vec_metadata(cx: &mut CrateContext,
                element_type: ty::t,
                span: Span)
             -> DICompositeType {

    let element_type_metadata = type_metadata(cx, element_type, span);
    let element_llvm_type = type_of::type_of(cx, element_type);
    let (element_size, element_align) = size_and_align_of(cx, element_llvm_type);

    let vec_llvm_type = Type::vec(cx.sess.targ_cfg.arch, &element_llvm_type);
    let vec_type_name: &str = format!("[{}]", ppaux::ty_to_str(cx.tcx, element_type));

    let member_llvm_types = vec_llvm_type.field_types();

    let int_type_metadata = type_metadata(cx, ty::mk_int(), span);
    let array_type_metadata = unsafe {
        llvm::LLVMDIBuilderCreateArrayType(
            DIB(cx),
            bytes_to_bits(element_size),
            bytes_to_bits(element_align),
            element_type_metadata,
            create_DIArray(DIB(cx), [llvm::LLVMDIBuilderGetOrCreateSubrange(DIB(cx), 0, 0)]))
    };

    let member_descriptions = [
        MemberDescription {
            name: @"fill",
            llvm_type: member_llvm_types[0],
            type_metadata: int_type_metadata,
            offset: ComputedMemberOffset,
        },
        MemberDescription {
            name: @"alloc",
            llvm_type: member_llvm_types[1],
            type_metadata: int_type_metadata,
            offset: ComputedMemberOffset,
        },
        MemberDescription {
            name: @"elements",
            llvm_type: member_llvm_types[2],
            type_metadata: array_type_metadata,
            offset: ComputedMemberOffset,
        }
    ];

    assert!(member_descriptions.len() == member_llvm_types.len());

    let loc = span_start(cx, span);
    let file_metadata = file_metadata(cx, loc.file.name);

    return composite_type_metadata(
        cx,
        vec_llvm_type,
        vec_type_name,
        member_descriptions,
        file_metadata,
        file_metadata,
        span);
}

fn boxed_vec_metadata(cx: &mut CrateContext,
                      element_type: ty::t,
                      span: Span)
                   -> DICompositeType {
    let element_llvm_type = type_of::type_of(cx, element_type);
    let vec_llvm_type = Type::vec(cx.sess.targ_cfg.arch, &element_llvm_type);
    let vec_type_name: &str = format!("[{}]", ppaux::ty_to_str(cx.tcx, element_type));
    let vec_metadata = vec_metadata(cx, element_type, span);

    return boxed_type_metadata(
        cx,
        Some(vec_type_name),
        vec_llvm_type,
        vec_metadata,
        span);
}

fn vec_slice_metadata(cx: &mut CrateContext,
                      vec_type: ty::t,
                      element_type: ty::t,
                      span: Span)
                   -> DICompositeType {

    debug!("vec_slice_metadata: {:?}", ty::get(vec_type));

    let slice_llvm_type = type_of::type_of(cx, vec_type);
    let slice_type_name = ppaux::ty_to_str(cx.tcx, vec_type);

    let member_llvm_types = slice_llvm_type.field_types();
    assert!(slice_layout_is_correct(cx, member_llvm_types, element_type));

    let data_ptr_type = ty::mk_ptr(cx.tcx, ty::mt { ty: element_type, mutbl: ast::MutImmutable });

    let member_descriptions = [
        MemberDescription {
            name: @"data_ptr",
            llvm_type: member_llvm_types[0],
            type_metadata: type_metadata(cx, data_ptr_type, span),
            offset: ComputedMemberOffset,
        },
        MemberDescription {
            name: @"length",
            llvm_type: member_llvm_types[1],
            type_metadata: type_metadata(cx, ty::mk_uint(), span),
            offset: ComputedMemberOffset,
        },
    ];

    assert!(member_descriptions.len() == member_llvm_types.len());

    let loc = span_start(cx, span);
    let file_metadata = file_metadata(cx, loc.file.name);

    return composite_type_metadata(
        cx,
        slice_llvm_type,
        slice_type_name,
        member_descriptions,
        file_metadata,
        file_metadata,
        span);

    fn slice_layout_is_correct(cx: &mut CrateContext,
                               member_llvm_types: &[Type],
                               element_type: ty::t)
                            -> bool {
        member_llvm_types.len() == 2 &&
        member_llvm_types[0] == type_of::type_of(cx, element_type).ptr_to() &&
        member_llvm_types[1] == cx.int_type
    }
}

fn subroutine_type_metadata(cx: &mut CrateContext,
                            signature: &ty::FnSig,
                            span: Span)
                         -> DICompositeType {
    let loc = span_start(cx, span);
    let file_metadata = file_metadata(cx, loc.file.name);

    let mut signature_metadata: ~[DIType] = vec::with_capacity(signature.inputs.len() + 1);

    // return type
    signature_metadata.push(match ty::get(signature.output).sty {
        ty::ty_nil => ptr::null(),
        _ => type_metadata(cx, signature.output, span)
    });

    // regular arguments
    for &argument_type in signature.inputs.iter() {
        signature_metadata.push(type_metadata(cx, argument_type, span));
    }

    return unsafe {
        llvm::LLVMDIBuilderCreateSubroutineType(
            DIB(cx),
            file_metadata,
            create_DIArray(DIB(cx), signature_metadata))
    };
}

fn trait_metadata(cx: &mut CrateContext,
                  def_id: ast::DefId,
                  trait_type: ty::t,
                  substs: &ty::substs,
                  trait_store: ty::TraitStore,
                  mutability: ast::Mutability,
                  _: &ty::BuiltinBounds,
                  usage_site_span: Span)
               -> DIType {
    // The implementation provided here is a stub. It makes sure that the trait type is
    // assigned the correct name, size, namespace, and source location. But it does not describe
    // the trait's methods.
    let path = ty::item_path(cx.tcx, def_id);
    let ident = path.last().ident();
    let name = ppaux::trait_store_to_str(cx.tcx, trait_store) +
               ppaux::mutability_to_str(mutability) +
               token::ident_to_str(&ident);
    // Add type and region parameters
    let name = ppaux::parameterized(cx.tcx, name, &substs.regions, substs.tps);

    let (containing_scope, definition_span) =
        get_namespace_and_span_for_item(cx, def_id, usage_site_span);

    let file_name = span_start(cx, definition_span).file.name;
    let file_metadata = file_metadata(cx, file_name);

    let trait_llvm_type = type_of::type_of(cx, trait_type);

    return composite_type_metadata(cx,
                                   trait_llvm_type,
                                   name,
                                   [],
                                   containing_scope,
                                   file_metadata,
                                   definition_span);
}

fn unimplemented_type_metadata(cx: &mut CrateContext, t: ty::t) -> DIType {
    debug!("unimplemented_type_metadata: {:?}", ty::get(t));

    let name = ppaux::ty_to_str(cx.tcx, t);
    let metadata = do format!("NYI<{}>", name).with_c_str |name| {
        unsafe {
            llvm::LLVMDIBuilderCreateBasicType(
                DIB(cx),
                name,
                0_u64,
                8_u64,
                DW_ATE_unsigned as c_uint)
            }
        };

    return metadata;
}

fn cache_id_for_type(t: ty::t) -> uint {
    ty::type_id(t)
}

fn type_metadata(cx: &mut CrateContext,
                 t: ty::t,
                 usage_site_span: Span)
              -> DIType {
    let cache_id = cache_id_for_type(t);
    match debug_context(cx).created_types.find(&cache_id) {
        Some(type_metadata) => return *type_metadata,
        None => ()
    }

    fn create_pointer_to_box_metadata(cx: &mut CrateContext,
                                      pointer_type: ty::t,
                                      type_in_box: ty::t)
                                   -> DIType {
        let content_type_name: &str = ppaux::ty_to_str(cx.tcx, type_in_box);
        let content_llvm_type = type_of::type_of(cx, type_in_box);
        let content_type_metadata = type_metadata(
            cx,
            type_in_box,
            codemap::dummy_sp());

        let box_metadata = boxed_type_metadata(
            cx,
            Some(content_type_name),
            content_llvm_type,
            content_type_metadata,
            codemap::dummy_sp());

        pointer_type_metadata(cx, pointer_type, box_metadata)
    }

    debug!("type_metadata: {:?}", ty::get(t));

    let sty = &ty::get(t).sty;
    let type_metadata = match *sty {
        ty::ty_nil      |
        ty::ty_bot      |
        ty::ty_bool     |
        ty::ty_char     |
        ty::ty_int(_)   |
        ty::ty_uint(_)  |
        ty::ty_float(_) => {
            basic_type_metadata(cx, t)
        },
        ty::ty_estr(ref vstore) => {
            let i8_t = ty::mk_i8();
            match *vstore {
                ty::vstore_fixed(len) => {
                    fixed_vec_metadata(cx, i8_t, len + 1, usage_site_span)
                },
                ty::vstore_uniq  => {
                    let vec_metadata = vec_metadata(cx, i8_t, usage_site_span);
                    pointer_type_metadata(cx, t, vec_metadata)
                }
                ty::vstore_box => {
                    let boxed_vec_metadata = boxed_vec_metadata(cx, i8_t, usage_site_span);
                    pointer_type_metadata(cx, t, boxed_vec_metadata)
                }
                ty::vstore_slice(_region) => {
                    vec_slice_metadata(cx, t, i8_t, usage_site_span)
                }
            }
        },
        ty::ty_enum(def_id, _) => {
            prepare_enum_metadata(cx, t, def_id, usage_site_span).finalize(cx)
        },
        ty::ty_box(ref mt) => {
            create_pointer_to_box_metadata(cx, t, mt.ty)
        },
        ty::ty_evec(ref mt, ref vstore) => {
            match *vstore {
                ty::vstore_fixed(len) => {
                    fixed_vec_metadata(cx, mt.ty, len, usage_site_span)
                }
                ty::vstore_uniq if ty::type_contents(cx.tcx, mt.ty).contains_managed() => {
                    let boxed_vec_metadata = boxed_vec_metadata(cx, mt.ty, usage_site_span);
                    pointer_type_metadata(cx, t, boxed_vec_metadata)
                }
                ty::vstore_uniq => {
                    let vec_metadata = vec_metadata(cx, mt.ty, usage_site_span);
                    pointer_type_metadata(cx, t, vec_metadata)
                }
                ty::vstore_box => {
                    let boxed_vec_metadata = boxed_vec_metadata(cx, mt.ty, usage_site_span);
                    pointer_type_metadata(cx, t, boxed_vec_metadata)
                }
                ty::vstore_slice(_) => {
                    vec_slice_metadata(cx, t, mt.ty, usage_site_span)
                }
            }
        },
        ty::ty_uniq(ref mt) if ty::type_contents(cx.tcx, mt.ty).contains_managed() => {
            create_pointer_to_box_metadata(cx, t, mt.ty)
        },
        ty::ty_uniq(ref mt)    |
        ty::ty_ptr(ref mt)     |
        ty::ty_rptr(_, ref mt) => {
            let pointee = type_metadata(cx, mt.ty, usage_site_span);
            pointer_type_metadata(cx, t, pointee)
        },
        ty::ty_bare_fn(ref barefnty) => {
            subroutine_type_metadata(cx, &barefnty.sig, usage_site_span)
        },
        ty::ty_closure(ref closurety) => {
            subroutine_type_metadata(cx, &closurety.sig, usage_site_span)
        },
        ty::ty_trait(def_id, ref substs, trait_store, mutability, ref bounds) => {
            trait_metadata(cx, def_id, t, substs, trait_store, mutability, bounds, usage_site_span)
        },
        ty::ty_struct(def_id, ref substs) => {
            prepare_struct_metadata(cx, t, def_id, substs, usage_site_span).finalize(cx)
        },
        ty::ty_tup(ref elements) => {
            prepare_tuple_metadata(cx, t, *elements, usage_site_span).finalize(cx)
        },
        ty::ty_opaque_box => {
            create_pointer_to_box_metadata(cx, t, ty::mk_nil())
        }
        _ => cx.sess.bug(format!("debuginfo: unexpected type in type_metadata: {:?}", sty))
    };

    debug_context(cx).created_types.insert(cache_id, type_metadata);
    return type_metadata;
}

#[deriving(Eq)]
enum DebugLocation {
    KnownLocation { scope: DIScope, line: uint, col: uint },
    UnknownLocation
}

impl DebugLocation {
    fn new(scope: DIScope, line: uint, col: uint) -> DebugLocation {
        KnownLocation {
            scope: scope,
            line: line,
            col: col,
        }
    }
}

fn set_debug_location(cx: &mut CrateContext, debug_location: DebugLocation) {
    if debug_location == debug_context(cx).current_debug_location {
        return;
    }

    let metadata_node;

    match debug_location {
        KnownLocation { scope, line, col } => {
            debug!("setting debug location to {} {}", line, col);
            let elements = [C_i32(line as i32), C_i32(col as i32), scope, ptr::null()];
            unsafe {
                metadata_node = llvm::LLVMMDNodeInContext(debug_context(cx).llcontext,
                                                          vec::raw::to_ptr(elements),
                                                          elements.len() as c_uint);
            }
        }
        UnknownLocation => {
            debug!("clearing debug location ");
            metadata_node = ptr::null();
        }
    };

    unsafe {
        llvm::LLVMSetCurrentDebugLocation(cx.builder.B, metadata_node);
    }

    debug_context(cx).current_debug_location = debug_location;
}

//=-------------------------------------------------------------------------------------------------
//  Utility Functions
//=-------------------------------------------------------------------------------------------------

#[inline]
fn roundup(x: uint, a: uint) -> uint {
    ((x + (a - 1)) / a) * a
}

/// Return codemap::Loc corresponding to the beginning of the span
fn span_start(cx: &CrateContext, span: Span) -> codemap::Loc {
    cx.sess.codemap.lookup_char_pos(span.lo)
}

fn size_and_align_of(cx: &mut CrateContext, llvm_type: Type) -> (uint, uint) {
    (machine::llsize_of_alloc(cx, llvm_type), machine::llalign_of_min(cx, llvm_type))
}

fn bytes_to_bits(bytes: uint) -> c_ulonglong {
    (bytes * 8) as c_ulonglong
}

#[inline]
fn debug_context<'a>(cx: &'a mut CrateContext) -> &'a mut CrateDebugContext {
    cx.dbg_cx.get_mut_ref()
}

#[inline]
fn DIB(cx: &CrateContext) -> DIBuilderRef {
    cx.dbg_cx.get_ref().builder
}

fn fn_should_be_ignored(fcx: &FunctionContext) -> bool {
    match fcx.debug_context {
        FunctionDebugContext(_) => false,
        _ => true
    }
}

fn assert_type_for_node_id(cx: &CrateContext, node_id: ast::NodeId, error_span: Span) {
    if !cx.tcx.node_types.contains_key(&(node_id as uint)) {
        cx.sess.span_bug(error_span, "debuginfo: Could not find type for node id!");
    }
}

fn get_namespace_and_span_for_item(cx: &mut CrateContext,
                                   def_id: ast::DefId,
                                   warning_span: Span)
                                -> (DIScope, Span) {
    let containing_scope = namespace_for_item(cx, def_id, warning_span).scope;
    let definition_span = if def_id.crate == ast::LOCAL_CRATE {
        let definition_span = match cx.tcx.items.find(&def_id.node) {
            Some(&ast_map::node_item(@ast::item { span, _ }, _)) => span,
            ref node => {
                cx.sess.span_warn(warning_span,
                    format!("debuginfo::get_namespace_and_span_for_item() \
                             - Unexpected node type: {:?}", *node));
                codemap::dummy_sp()
            }
        };
        definition_span
    } else {
        // For external items there is no span information
        codemap::dummy_sp()
    };

    (containing_scope, definition_span)
}

// This procedure builds the *scope map* for a given function, which maps any given ast::NodeId in
// the function's AST to the correct DIScope metadata instance.
//
// This builder procedure walks the AST in execution order and keeps track of what belongs to which
// scope, creating DIScope DIEs along the way, and introducing *artificial* lexical scope
// descriptors where necessary. These artificial scopes allow GDB to correctly handle name
// shadowing.
fn populate_scope_map(cx: &mut CrateContext,
                      arg_pats: &[@ast::Pat],
                      fn_entry_block: &ast::Block,
                      fn_metadata: DISubprogram,
                      scope_map: &mut HashMap<ast::NodeId, DIScope>) {
    let def_map = cx.tcx.def_map;

    struct ScopeStackEntry {
        scope_metadata: DIScope,
        ident: Option<ast::Ident>
    }

    let mut scope_stack = ~[ScopeStackEntry { scope_metadata: fn_metadata, ident: None }];

    // Push argument identifiers onto the stack so arguments integrate nicely with variable
    // shadowing.
    for &arg_pat in arg_pats.iter() {
        do pat_util::pat_bindings(def_map, arg_pat) |_, _, _, path_ref| {
            let ident = ast_util::path_to_ident(path_ref);
            scope_stack.push(ScopeStackEntry { scope_metadata: fn_metadata, ident: Some(ident) });
        }
    }

    walk_block(cx, fn_entry_block, &mut scope_stack, scope_map);

    // local helper functions for walking the AST.
    fn with_new_scope(cx: &mut CrateContext,
                      scope_span: Span,
                      scope_stack: &mut ~[ScopeStackEntry],
                      scope_map: &mut HashMap<ast::NodeId, DIScope>,
                      inner_walk: &fn(&mut CrateContext,
                                      &mut ~[ScopeStackEntry],
                                      &mut HashMap<ast::NodeId, DIScope>)) {
        // Create a new lexical scope and push it onto the stack
        let loc = cx.sess.codemap.lookup_char_pos(scope_span.lo);
        let file_metadata = file_metadata(cx, loc.file.name);
        let parent_scope = scope_stack.last().scope_metadata;

        let scope_metadata = unsafe {
            llvm::LLVMDIBuilderCreateLexicalBlock(
                DIB(cx),
                parent_scope,
                file_metadata,
                loc.line as c_uint,
                loc.col.to_uint() as c_uint)
        };

        scope_stack.push(ScopeStackEntry { scope_metadata: scope_metadata, ident: None });

        inner_walk(cx, scope_stack, scope_map);

        // pop artificial scopes
        while scope_stack.last().ident.is_some() {
            scope_stack.pop();
        }

        if scope_stack.last().scope_metadata != scope_metadata {
            cx.sess.span_bug(scope_span, "debuginfo: Inconsistency in scope management.");
        }

        scope_stack.pop();
    }

    fn walk_block(cx: &mut CrateContext,
                  block: &ast::Block,
                  scope_stack: &mut ~[ScopeStackEntry],
                  scope_map: &mut HashMap<ast::NodeId, DIScope>) {
        scope_map.insert(block.id, scope_stack.last().scope_metadata);

        // The interesting things here are statements and the concluding expression.
        for &@ ref statement in block.stmts.iter() {
            scope_map.insert(ast_util::stmt_id(statement), scope_stack.last().scope_metadata);

            match statement.node {
                ast::StmtDecl(@ref decl, _) => walk_decl(cx, decl, scope_stack, scope_map),
                ast::StmtExpr(@ref exp, _) |
                ast::StmtSemi(@ref exp, _) => walk_expr(cx, exp, scope_stack, scope_map),
                ast::StmtMac(*) => () // ignore macros (which should be expanded anyway)
            }
        }

        for &@ref exp in block.expr.iter() {
            walk_expr(cx, exp, scope_stack, scope_map);
        }
    }

    fn walk_decl(cx: &mut CrateContext,
                 decl: &ast::Decl,
                 scope_stack: &mut ~[ScopeStackEntry],
                 scope_map: &mut HashMap<ast::NodeId, DIScope>) {
        match *decl {
            codemap::Spanned { node: ast::DeclLocal(@ref local), _ } => {
                scope_map.insert(local.id, scope_stack.last().scope_metadata);

                walk_pattern(cx, local.pat, scope_stack, scope_map);

                for &@ref exp in local.init.iter() {
                    walk_expr(cx, exp, scope_stack, scope_map);
                }
            }
            _ => ()
        }
    }

    fn walk_pattern(cx: &mut CrateContext,
                    pat: @ast::Pat,
                    scope_stack: &mut ~[ScopeStackEntry],
                    scope_map: &mut HashMap<ast::NodeId, DIScope>) {

        let def_map = cx.tcx.def_map;

        // Unfortunately, we cannot just use pat_util::pat_bindings() or ast_util::walk_pat() here
        // because we have to visit *all* nodes in order to put them into the scope map. The above
        // functions don't do that.
        match pat.node {
            ast::PatIdent(_, ref path_ref, ref sub_pat_opt) => {

                // Check if this is a binding. If so we need to put it on the scope stack and maybe
                // introduce an articial scope
                if pat_util::pat_is_binding(def_map, pat) {

                    let ident = ast_util::path_to_ident(path_ref);

                    // LLVM does not properly generate 'DW_AT_start_scope' fields for variable DIEs.
                    // For this reason we have to introduce an artificial scope at bindings whenever
                    // a variable with the same name is declared in *any* parent scope.
                    //
                    // Otherwise the following error occurs:
                    //
                    // let x = 10;
                    //
                    // do_something(); // 'gdb print x' correctly prints 10
                    //
                    // {
                    //     do_something(); // 'gdb print x' prints 0, because it already reads the
                    //                     // uninitialized 'x' from the next line...
                    //     let x = 100;
                    //     do_something(); // 'gdb print x' correctly prints 100
                    // }

                    // Is there already a binding with that name?
                    // N.B.: this comparison must be UNhygienic... because
                    // gdb knows nothing about the context, so any two
                    // variables with the same name will cause the problem.
                    let need_new_scope = scope_stack
                        .iter()
                        .any(|entry| entry.ident.iter().any(|i| i.name == ident.name));

                    if need_new_scope {
                        // Create a new lexical scope and push it onto the stack
                        let loc = cx.sess.codemap.lookup_char_pos(pat.span.lo);
                        let file_metadata = file_metadata(cx, loc.file.name);
                        let parent_scope = scope_stack.last().scope_metadata;

                        let scope_metadata = unsafe {
                            llvm::LLVMDIBuilderCreateLexicalBlock(
                                DIB(cx),
                                parent_scope,
                                file_metadata,
                                loc.line as c_uint,
                                loc.col.to_uint() as c_uint)
                        };

                        scope_stack.push(ScopeStackEntry {
                            scope_metadata: scope_metadata,
                            ident: Some(ident)
                        });

                    } else {
                        // Push a new entry anyway so the name can be found
                        let prev_metadata = scope_stack.last().scope_metadata;
                        scope_stack.push(ScopeStackEntry {
                            scope_metadata: prev_metadata,
                            ident: Some(ident)
                        });
                    }
                }

                scope_map.insert(pat.id, scope_stack.last().scope_metadata);

                for &sub_pat in sub_pat_opt.iter() {
                    walk_pattern(cx, sub_pat, scope_stack, scope_map);
                }
            }

            ast::PatWild => {
                scope_map.insert(pat.id, scope_stack.last().scope_metadata);
            }

            ast::PatEnum(_, ref sub_pats_opt) => {
                scope_map.insert(pat.id, scope_stack.last().scope_metadata);

                for ref sub_pats in sub_pats_opt.iter() {
                    for &p in sub_pats.iter() {
                        walk_pattern(cx, p, scope_stack, scope_map);
                    }
                }
            }

            ast::PatStruct(_, ref field_pats, _) => {
                scope_map.insert(pat.id, scope_stack.last().scope_metadata);

                for &ast::FieldPat { pat: sub_pat, _ } in field_pats.iter() {
                    walk_pattern(cx, sub_pat, scope_stack, scope_map);
                }
            }

            ast::PatTup(ref sub_pats) => {
                scope_map.insert(pat.id, scope_stack.last().scope_metadata);

                for &sub_pat in sub_pats.iter() {
                    walk_pattern(cx, sub_pat, scope_stack, scope_map);
                }
            }

            ast::PatBox(sub_pat)    |
            ast::PatUniq(sub_pat)   |
            ast::PatRegion(sub_pat) => {
                scope_map.insert(pat.id, scope_stack.last().scope_metadata);
                walk_pattern(cx, sub_pat, scope_stack, scope_map);
            }

            ast::PatLit(@ref exp) => {
                scope_map.insert(pat.id, scope_stack.last().scope_metadata);
                walk_expr(cx, exp, scope_stack, scope_map);
            }

            ast::PatRange(@ref exp1, @ref exp2) => {
                scope_map.insert(pat.id, scope_stack.last().scope_metadata);
                walk_expr(cx, exp1, scope_stack, scope_map);
                walk_expr(cx, exp2, scope_stack, scope_map);
            }

            ast::PatVec(ref front_sub_pats, ref middle_sub_pats, ref back_sub_pats) => {
                scope_map.insert(pat.id, scope_stack.last().scope_metadata);

                for &sub_pat in front_sub_pats.iter() {
                    walk_pattern(cx, sub_pat, scope_stack, scope_map);
                }

                for &sub_pat in middle_sub_pats.iter() {
                    walk_pattern(cx, sub_pat, scope_stack, scope_map);
                }

                for &sub_pat in back_sub_pats.iter() {
                    walk_pattern(cx, sub_pat, scope_stack, scope_map);
                }
            }
        }
    }

    fn walk_expr(cx: &mut CrateContext,
                 exp: &ast::Expr,
                 scope_stack: &mut ~[ScopeStackEntry],
                 scope_map: &mut HashMap<ast::NodeId, DIScope>) {

        scope_map.insert(exp.id, scope_stack.last().scope_metadata);

        match exp.node {
            ast::ExprLogLevel |
            ast::ExprSelf     |
            ast::ExprLit(_)   |
            ast::ExprBreak(_) |
            ast::ExprAgain(_) |
            ast::ExprPath(_)  => (),

            ast::ExprVstore(@ref sub_exp, _)   |
            ast::ExprCast(@ref sub_exp, _)     |
            ast::ExprAddrOf(_, @ref sub_exp)  |
            ast::ExprField(@ref sub_exp, _, _) |
            ast::ExprParen(@ref sub_exp)       => walk_expr(cx, sub_exp, scope_stack, scope_map),

            ast::ExprRet(exp_opt) => match exp_opt {
                Some(@ref sub_exp) => walk_expr(cx, sub_exp, scope_stack, scope_map),
                None => ()
            },

            ast::ExprUnary(node_id, _, @ref sub_exp) => {
                scope_map.insert(node_id, scope_stack.last().scope_metadata);
                walk_expr(cx, sub_exp, scope_stack, scope_map);
            }

            ast::ExprAssignOp(node_id, _, @ref lhs, @ref rhs) |
            ast::ExprIndex(node_id, @ref lhs, @ref rhs)        |
            ast::ExprBinary(node_id, _, @ref lhs, @ref rhs)    => {
                scope_map.insert(node_id, scope_stack.last().scope_metadata);
                walk_expr(cx, lhs, scope_stack, scope_map);
                walk_expr(cx, rhs, scope_stack, scope_map);
            }

            ast::ExprVec(ref init_expressions, _) |
            ast::ExprTup(ref init_expressions)    => {
                for &@ref ie in init_expressions.iter() {
                    walk_expr(cx, ie, scope_stack, scope_map);
                }
            }

            ast::ExprAssign(@ref sub_exp1, @ref sub_exp2)    |
            ast::ExprRepeat(@ref sub_exp1, @ref sub_exp2, _) => {
                walk_expr(cx, sub_exp1, scope_stack, scope_map);
                walk_expr(cx, sub_exp2, scope_stack, scope_map);
            }

            ast::ExprIf(@ref cond_exp, ref then_block, ref opt_else_exp) => {
                walk_expr(cx, cond_exp, scope_stack, scope_map);

                do with_new_scope(cx, then_block.span, scope_stack, scope_map) |cx,
                                                                                scope_stack,
                                                                                scope_map| {
                    walk_block(cx, then_block, scope_stack, scope_map);
                }

                match *opt_else_exp {
                    Some(@ref else_exp) => walk_expr(cx, else_exp, scope_stack, scope_map),
                    _ => ()
                }
            }

            ast::ExprWhile(@ref cond_exp, ref loop_body) => {
                walk_expr(cx, cond_exp, scope_stack, scope_map);

                do with_new_scope(cx, loop_body.span, scope_stack, scope_map) |cx,
                                                                               scope_stack,
                                                                               scope_map| {
                    walk_block(cx, loop_body, scope_stack, scope_map);
                }
            }

            ast::ExprForLoop(_, _, _, _) => {
                cx.sess.span_bug(exp.span, "debuginfo::populate_scope_map() - \
                                            Found unexpanded for-loop.");
            }

            ast::ExprMac(_) => {
                cx.sess.span_bug(exp.span, "debuginfo::populate_scope_map() - \
                                            Found unexpanded macro.");
            }

            ast::ExprLoop(ref block, _) |
            ast::ExprBlock(ref block)   => {
                do with_new_scope(cx, block.span, scope_stack, scope_map) |cx,
                                                                           scope_stack,
                                                                           scope_map| {
                    walk_block(cx, block, scope_stack, scope_map);
                }
            }

            ast::ExprFnBlock(ast::fn_decl { inputs: ref inputs, _ }, ref block) |
            ast::ExprProc(ast::fn_decl { inputs: ref inputs, _ }, ref block) => {
                do with_new_scope(cx, block.span, scope_stack, scope_map) |cx,
                                                                           scope_stack,
                                                                           scope_map| {
                    for &ast::arg { pat: pattern, _ } in inputs.iter() {
                        walk_pattern(cx, pattern, scope_stack, scope_map);
                    }

                    walk_block(cx, block, scope_stack, scope_map);
                }
            }

            // ast::expr_loop_body(@ref inner_exp) |
            ast::ExprDoBody(@ref inner_exp)   => {
                let inner_expr_is_expr_fn_block = match *inner_exp {
                    ast::Expr { node: ast::ExprFnBlock(*), _ } => true,
                    _ => false
                };

                if !inner_expr_is_expr_fn_block {
                    cx.sess.span_bug(inner_exp.span, "debuginfo: Inner expression was expected \
                                                      to be an ast::expr_fn_block.");
                }

                walk_expr(cx, inner_exp, scope_stack, scope_map);
            }

            ast::ExprCall(@ref fn_exp, ref args, _) => {
                walk_expr(cx, fn_exp, scope_stack, scope_map);

                for &@ref arg_exp in args.iter() {
                    walk_expr(cx, arg_exp, scope_stack, scope_map);
                }
            }

            ast::ExprMethodCall(node_id, @ref receiver_exp, _, _, ref args, _) => {
                scope_map.insert(node_id, scope_stack.last().scope_metadata);
                walk_expr(cx, receiver_exp, scope_stack, scope_map);

                for &@ref arg_exp in args.iter() {
                    walk_expr(cx, arg_exp, scope_stack, scope_map);
                }
            }

            ast::ExprMatch(@ref discriminant_exp, ref arms) => {
                walk_expr(cx, discriminant_exp, scope_stack, scope_map);

                // for each arm we have to first walk the pattern as these might introduce new
                // artificial scopes. It should be sufficient to walk only one pattern per arm, as
                // they all must contain the same binding names

                for arm_ref in arms.iter() {
                    let arm_span = arm_ref.pats[0].span;

                    do with_new_scope(cx, arm_span, scope_stack, scope_map) |cx,
                                                                             scope_stack,
                                                                             scope_map| {
                        for &pat in arm_ref.pats.iter() {
                            walk_pattern(cx, pat, scope_stack, scope_map);
                        }

                        for &@ref guard_exp in arm_ref.guard.iter() {
                            walk_expr(cx, guard_exp, scope_stack, scope_map)
                        }

                        walk_block(cx, &arm_ref.body, scope_stack, scope_map);
                    }
                }
            }

            ast::ExprStruct(_, ref fields, ref base_exp) => {
                for &ast::Field { expr: @ref exp, _ } in fields.iter() {
                    walk_expr(cx, exp, scope_stack, scope_map);
                }

                match *base_exp {
                    Some(@ref exp) => walk_expr(cx, exp, scope_stack, scope_map),
                    None => ()
                }
            }

            ast::ExprInlineAsm(ast::inline_asm { inputs: ref inputs,
                                                   outputs: ref outputs,
                                                   _ }) => {
                // inputs, outputs: ~[(@str, @expr)]
                for &(_, @ref exp) in inputs.iter() {
                    walk_expr(cx, exp, scope_stack, scope_map);
                }

                for &(_, @ref exp) in outputs.iter() {
                    walk_expr(cx, exp, scope_stack, scope_map);
                }
            }
        }
    }
}


//=-------------------------------------------------------------------------------------------------
// Namespace Handling
//=-------------------------------------------------------------------------------------------------

struct NamespaceTreeNode {
    ident: ast::Ident,
    scope: DIScope,
    parent: Option<@NamespaceTreeNode>,
}

impl NamespaceTreeNode {
    fn mangled_name_of_contained_item(&self, item_name: &str) -> ~str {
        let mut name = ~"_ZN";
        fill_nested(self, &mut name);

        name.push_str(format!("{}{}", item_name.len(), item_name));
        name.push_char('E');

        return name;

        fn fill_nested(node: &NamespaceTreeNode, output: &mut ~str) {
            match node.parent {
                Some(parent) => {
                    fill_nested(parent, output);
                }
                None => {}
            }
            let name = token::ident_to_str(&node.ident);
            output.push_str(format!("{}{}", name.len(), name));
        }
    }
}

fn namespace_for_item(cx: &mut CrateContext,
                      def_id: ast::DefId,
                      warning_span: Span)
                   -> @NamespaceTreeNode {
    let namespace_path = {
        let mut item_path = ty::item_path(cx.tcx, def_id);

        if (def_id.crate == ast::LOCAL_CRATE && item_path.len() < 1) ||
           (def_id.crate != ast::LOCAL_CRATE && item_path.len() < 2) {
            cx.sess.bug(format!("debuginfo::namespace_for_item() - Item path too short: {}",
                ast_map::path_to_str(item_path, token::get_ident_interner())));
        }

        // remove the name of the item
        item_path.pop();

        if def_id.crate == ast::LOCAL_CRATE {
            // prepend crate name if not already present
            let crate_namespace_ident = token::str_to_ident(cx.link_meta.name);
            item_path.insert(0, ast_map::path_mod(crate_namespace_ident));
        }

        item_path
    };

    let mut current_key = vec::with_capacity(namespace_path.len());
    let mut parent_node: Option<@NamespaceTreeNode> = None;
    let last_index = namespace_path.len() - 1;

    // Create/Lookup namespace for each element of the path.
    for (i, &path_element) in namespace_path.iter().enumerate() {
        let ident = path_element.ident();
        current_key.push(ident);

        let existing_node = debug_context(cx).namespace_map.find_copy(&current_key);
        let current_node = match existing_node {
            Some(existing_node) => existing_node,
            None => {
                // create and insert
                let parent_scope = match parent_node {
                    Some(node) => node.scope,
                    None => ptr::null()
                };
                let namespace_name = token::ident_to_str(&ident);

                let namespace_metadata = unsafe {
                    do namespace_name.with_c_str |namespace_name| {
                        llvm::LLVMDIBuilderCreateNameSpace(
                            DIB(cx),
                            parent_scope,
                            namespace_name,
                            ptr::null(), // cannot reconstruct file ...
                            0)           // ... or line information, but that's not so important.
                    }
                };

                let node = @NamespaceTreeNode {
                    ident: ident,
                    scope: namespace_metadata,
                    parent: parent_node,
                };

                debug_context(cx).namespace_map.insert(current_key.clone(), node);

                node
            }
        };

        if i == last_index {
            return current_node;
        } else {
            parent_node = Some(current_node);
        }
    }

    // Should be unreachable:
    let error_message = format!("debuginfo::namespace_for_item() - Code path should be \
                                 unreachable. namespace_path was {}",
                                 ast_map::path_to_str(namespace_path, token::get_ident_interner()));
    cx.sess.span_bug(warning_span, error_message);
}
