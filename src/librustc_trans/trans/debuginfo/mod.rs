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
use self::InternalDebugLocation::*;
use self::RecursiveTypeDescription::*;

use self::utils::{debug_context, DIB, span_start,
                  assert_type_for_node_id, fn_should_be_ignored,
                  contains_nodebug_attribute, create_scope_map};
use self::create::{declare_local, create_DIArray, is_node_local_to_unit};
use self::namespace::{namespace_for_item, NamespaceTreeNode};
use self::types::{compute_debuginfo_type_name, push_debuginfo_type_name};
use self::metadata::{type_metadata, file_metadata, scope_metadata,
                     compile_unit_metadata, MetadataCreationResult};
use self::adt::{MemberDescriptionFactory, set_members_of_composite_type};

use llvm;
use llvm::{ModuleRef, ContextRef, ValueRef};
use llvm::debuginfo::*;
use middle::subst::{self, Substs};
use trans::machine;
use trans::common::{self, NodeIdAndSpan, CrateContext, FunctionContext, Block,
                    NormalizingClosureTyper};
use trans::_match::{BindingInfo, TrByCopy, TrByMove, TrByRef};
use trans::monomorphize;
use trans::type_::Type;
use middle::ty::{self, Ty, ClosureTyper};
use middle::pat_util;
use session::config::{self, FullDebugInfo, LimitedDebugInfo, NoDebugInfo};
use util::nodemap::{DefIdMap, NodeMap, FnvHashMap, FnvHashSet};
use util::ppaux;

use libc::c_uint;
use std::cell::{Cell, RefCell};
use std::ffi::CString;
use std::ptr;
use std::rc::Rc;
use syntax::util::interner::Interner;
use syntax::codemap::{Span, Pos};
use syntax::{ast, codemap, ast_util, ast_map};
use syntax::parse::token::{self, special_idents};

pub mod gdb;
mod utils;
mod create;
mod namespace;
mod types;
mod metadata;
mod adt;


#[allow(non_upper_case_globals)]
const DW_TAG_auto_variable: c_uint = 0x100;
#[allow(non_upper_case_globals)]
const DW_TAG_arg_variable: c_uint = 0x101;

const UNKNOWN_LINE_NUMBER: c_uint = 0;
const UNKNOWN_COLUMN_NUMBER: c_uint = 0;

// ptr::null() doesn't work :(
const UNKNOWN_FILE_METADATA: DIFile = (0 as DIFile);
const UNKNOWN_SCOPE_METADATA: DIScope = (0 as DIScope);

const FLAGS_NONE: c_uint = 0;

//=-----------------------------------------------------------------------------
//  Public Interface of debuginfo module
//=-----------------------------------------------------------------------------

#[derive(Copy, Debug, Hash, Eq, PartialEq, Clone)]
pub struct UniqueTypeId(ast::Name);

// The TypeMap is where the CrateDebugContext holds the type metadata nodes
// created so far. The metadata nodes are indexed by UniqueTypeId, and, for
// faster lookup, also by Ty. The TypeMap is responsible for creating
// UniqueTypeIds.
struct TypeMap<'tcx> {
    // The UniqueTypeIds created so far
    unique_id_interner: Interner<Rc<String>>,
    // A map from UniqueTypeId to debuginfo metadata for that type. This is a 1:1 mapping.
    unique_id_to_metadata: FnvHashMap<UniqueTypeId, DIType>,
    // A map from types to debuginfo metadata. This is a N:1 mapping.
    type_to_metadata: FnvHashMap<Ty<'tcx>, DIType>,
    // A map from types to UniqueTypeId. This is a N:1 mapping.
    type_to_unique_id: FnvHashMap<Ty<'tcx>, UniqueTypeId>
}

impl<'tcx> TypeMap<'tcx> {

    fn new() -> TypeMap<'tcx> {
        TypeMap {
            unique_id_interner: Interner::new(),
            type_to_metadata: FnvHashMap(),
            unique_id_to_metadata: FnvHashMap(),
            type_to_unique_id: FnvHashMap(),
        }
    }

    // Adds a Ty to metadata mapping to the TypeMap. The method will fail if
    // the mapping already exists.
    fn register_type_with_metadata<'a>(&mut self,
                                       cx: &CrateContext<'a, 'tcx>,
                                       type_: Ty<'tcx>,
                                       metadata: DIType) {
        if self.type_to_metadata.insert(type_, metadata).is_some() {
            cx.sess().bug(&format!("Type metadata for Ty '{}' is already in the TypeMap!",
                                   ppaux::ty_to_string(cx.tcx(), type_)));
        }
    }

    // Adds a UniqueTypeId to metadata mapping to the TypeMap. The method will
    // fail if the mapping already exists.
    fn register_unique_id_with_metadata(&mut self,
                                        cx: &CrateContext,
                                        unique_type_id: UniqueTypeId,
                                        metadata: DIType) {
        if self.unique_id_to_metadata.insert(unique_type_id, metadata).is_some() {
            let unique_type_id_str = self.get_unique_type_id_as_string(unique_type_id);
            cx.sess().bug(&format!("Type metadata for unique id '{}' is already in the TypeMap!",
                                  &unique_type_id_str[..]));
        }
    }

    fn find_metadata_for_type(&self, type_: Ty<'tcx>) -> Option<DIType> {
        self.type_to_metadata.get(&type_).cloned()
    }

    fn find_metadata_for_unique_id(&self, unique_type_id: UniqueTypeId) -> Option<DIType> {
        self.unique_id_to_metadata.get(&unique_type_id).cloned()
    }

    // Get the string representation of a UniqueTypeId. This method will fail if
    // the id is unknown.
    fn get_unique_type_id_as_string(&self, unique_type_id: UniqueTypeId) -> Rc<String> {
        let UniqueTypeId(interner_key) = unique_type_id;
        self.unique_id_interner.get(interner_key)
    }

    // Get the UniqueTypeId for the given type. If the UniqueTypeId for the given
    // type has been requested before, this is just a table lookup. Otherwise an
    // ID will be generated and stored for later lookup.
    fn get_unique_type_id_of_type<'a>(&mut self, cx: &CrateContext<'a, 'tcx>,
                                      type_: Ty<'tcx>) -> UniqueTypeId {

        // basic type           -> {:name of the type:}
        // tuple                -> {tuple_(:param-uid:)*}
        // struct               -> {struct_:svh: / :node-id:_<(:param-uid:),*> }
        // enum                 -> {enum_:svh: / :node-id:_<(:param-uid:),*> }
        // enum variant         -> {variant_:variant-name:_:enum-uid:}
        // reference (&)        -> {& :pointee-uid:}
        // mut reference (&mut) -> {&mut :pointee-uid:}
        // ptr (*)              -> {* :pointee-uid:}
        // mut ptr (*mut)       -> {*mut :pointee-uid:}
        // unique ptr (~)       -> {~ :pointee-uid:}
        // @-ptr (@)            -> {@ :pointee-uid:}
        // sized vec ([T; x])   -> {[:size:] :element-uid:}
        // unsized vec ([T])    -> {[] :element-uid:}
        // trait (T)            -> {trait_:svh: / :node-id:_<(:param-uid:),*> }
        // closure              -> {<unsafe_> <once_> :store-sigil: |(:param-uid:),* <,_...>| -> \
        //                             :return-type-uid: : (:bounds:)*}
        // function             -> {<unsafe_> <abi_> fn( (:param-uid:)* <,_...> ) -> \
        //                             :return-type-uid:}
        // unique vec box (~[]) -> {HEAP_VEC_BOX<:pointee-uid:>}
        // gc box               -> {GC_BOX<:pointee-uid:>}

        match self.type_to_unique_id.get(&type_).cloned() {
            Some(unique_type_id) => return unique_type_id,
            None => { /* generate one */}
        };

        let mut unique_type_id = String::with_capacity(256);
        unique_type_id.push('{');

        match type_.sty {
            ty::ty_bool     |
            ty::ty_char     |
            ty::ty_str      |
            ty::ty_int(_)   |
            ty::ty_uint(_)  |
            ty::ty_float(_) => {
                push_debuginfo_type_name(cx, type_, false, &mut unique_type_id);
            },
            ty::ty_enum(def_id, substs) => {
                unique_type_id.push_str("enum ");
                from_def_id_and_substs(self, cx, def_id, substs, &mut unique_type_id);
            },
            ty::ty_struct(def_id, substs) => {
                unique_type_id.push_str("struct ");
                from_def_id_and_substs(self, cx, def_id, substs, &mut unique_type_id);
            },
            ty::ty_tup(ref component_types) if component_types.is_empty() => {
                push_debuginfo_type_name(cx, type_, false, &mut unique_type_id);
            },
            ty::ty_tup(ref component_types) => {
                unique_type_id.push_str("tuple ");
                for &component_type in component_types {
                    let component_type_id =
                        self.get_unique_type_id_of_type(cx, component_type);
                    let component_type_id =
                        self.get_unique_type_id_as_string(component_type_id);
                    unique_type_id.push_str(&component_type_id[..]);
                }
            },
            ty::ty_uniq(inner_type) => {
                unique_type_id.push('~');
                let inner_type_id = self.get_unique_type_id_of_type(cx, inner_type);
                let inner_type_id = self.get_unique_type_id_as_string(inner_type_id);
                unique_type_id.push_str(&inner_type_id[..]);
            },
            ty::ty_ptr(ty::mt { ty: inner_type, mutbl } ) => {
                unique_type_id.push('*');
                if mutbl == ast::MutMutable {
                    unique_type_id.push_str("mut");
                }

                let inner_type_id = self.get_unique_type_id_of_type(cx, inner_type);
                let inner_type_id = self.get_unique_type_id_as_string(inner_type_id);
                unique_type_id.push_str(&inner_type_id[..]);
            },
            ty::ty_rptr(_, ty::mt { ty: inner_type, mutbl }) => {
                unique_type_id.push('&');
                if mutbl == ast::MutMutable {
                    unique_type_id.push_str("mut");
                }

                let inner_type_id = self.get_unique_type_id_of_type(cx, inner_type);
                let inner_type_id = self.get_unique_type_id_as_string(inner_type_id);
                unique_type_id.push_str(&inner_type_id[..]);
            },
            ty::ty_vec(inner_type, optional_length) => {
                match optional_length {
                    Some(len) => {
                        unique_type_id.push_str(&format!("[{}]", len));
                    }
                    None => {
                        unique_type_id.push_str("[]");
                    }
                };

                let inner_type_id = self.get_unique_type_id_of_type(cx, inner_type);
                let inner_type_id = self.get_unique_type_id_as_string(inner_type_id);
                unique_type_id.push_str(&inner_type_id[..]);
            },
            ty::ty_trait(ref trait_data) => {
                unique_type_id.push_str("trait ");

                let principal =
                    ty::erase_late_bound_regions(cx.tcx(),
                                                 &trait_data.principal);

                from_def_id_and_substs(self,
                                       cx,
                                       principal.def_id,
                                       principal.substs,
                                       &mut unique_type_id);
            },
            ty::ty_bare_fn(_, &ty::BareFnTy{ unsafety, abi, ref sig } ) => {
                if unsafety == ast::Unsafety::Unsafe {
                    unique_type_id.push_str("unsafe ");
                }

                unique_type_id.push_str(abi.name());

                unique_type_id.push_str(" fn(");

                let sig = ty::erase_late_bound_regions(cx.tcx(), sig);

                for &parameter_type in &sig.inputs {
                    let parameter_type_id =
                        self.get_unique_type_id_of_type(cx, parameter_type);
                    let parameter_type_id =
                        self.get_unique_type_id_as_string(parameter_type_id);
                    unique_type_id.push_str(&parameter_type_id[..]);
                    unique_type_id.push(',');
                }

                if sig.variadic {
                    unique_type_id.push_str("...");
                }

                unique_type_id.push_str(")->");
                match sig.output {
                    ty::FnConverging(ret_ty) => {
                        let return_type_id = self.get_unique_type_id_of_type(cx, ret_ty);
                        let return_type_id = self.get_unique_type_id_as_string(return_type_id);
                        unique_type_id.push_str(&return_type_id[..]);
                    }
                    ty::FnDiverging => {
                        unique_type_id.push_str("!");
                    }
                }
            },
            ty::ty_closure(def_id, substs) => {
                let typer = NormalizingClosureTyper::new(cx.tcx());
                let closure_ty = typer.closure_type(def_id, substs);
                self.get_unique_type_id_of_closure_type(cx,
                                                        closure_ty,
                                                        &mut unique_type_id);
            },
            _ => {
                cx.sess().bug(&format!("get_unique_type_id_of_type() - unexpected type: {}, {:?}",
                                      &ppaux::ty_to_string(cx.tcx(), type_),
                                      type_.sty))
            }
        };

        unique_type_id.push('}');

        // Trim to size before storing permanently
        unique_type_id.shrink_to_fit();

        let key = self.unique_id_interner.intern(Rc::new(unique_type_id));
        self.type_to_unique_id.insert(type_, UniqueTypeId(key));

        return UniqueTypeId(key);

        fn from_def_id_and_substs<'a, 'tcx>(type_map: &mut TypeMap<'tcx>,
                                            cx: &CrateContext<'a, 'tcx>,
                                            def_id: ast::DefId,
                                            substs: &subst::Substs<'tcx>,
                                            output: &mut String) {
            // First, find out the 'real' def_id of the type. Items inlined from
            // other crates have to be mapped back to their source.
            let source_def_id = if def_id.krate == ast::LOCAL_CRATE {
                match cx.external_srcs().borrow().get(&def_id.node).cloned() {
                    Some(source_def_id) => {
                        // The given def_id identifies the inlined copy of a
                        // type definition, let's take the source of the copy.
                        source_def_id
                    }
                    None => def_id
                }
            } else {
                def_id
            };

            // Get the crate hash as first part of the identifier.
            let crate_hash = if source_def_id.krate == ast::LOCAL_CRATE {
                cx.link_meta().crate_hash.clone()
            } else {
                cx.sess().cstore.get_crate_hash(source_def_id.krate)
            };

            output.push_str(crate_hash.as_str());
            output.push_str("/");
            output.push_str(&format!("{:x}", def_id.node));

            // Maybe check that there is no self type here.

            let tps = substs.types.get_slice(subst::TypeSpace);
            if !tps.is_empty() {
                output.push('<');

                for &type_parameter in tps {
                    let param_type_id =
                        type_map.get_unique_type_id_of_type(cx, type_parameter);
                    let param_type_id =
                        type_map.get_unique_type_id_as_string(param_type_id);
                    output.push_str(&param_type_id[..]);
                    output.push(',');
                }

                output.push('>');
            }
        }
    }

    fn get_unique_type_id_of_closure_type<'a>(&mut self,
                                              cx: &CrateContext<'a, 'tcx>,
                                              closure_ty: ty::ClosureTy<'tcx>,
                                              unique_type_id: &mut String) {
        let ty::ClosureTy { unsafety,
                            ref sig,
                            abi: _ } = closure_ty;

        if unsafety == ast::Unsafety::Unsafe {
            unique_type_id.push_str("unsafe ");
        }

        unique_type_id.push_str("|");

        let sig = ty::erase_late_bound_regions(cx.tcx(), sig);

        for &parameter_type in &sig.inputs {
            let parameter_type_id =
                self.get_unique_type_id_of_type(cx, parameter_type);
            let parameter_type_id =
                self.get_unique_type_id_as_string(parameter_type_id);
            unique_type_id.push_str(&parameter_type_id[..]);
            unique_type_id.push(',');
        }

        if sig.variadic {
            unique_type_id.push_str("...");
        }

        unique_type_id.push_str("|->");

        match sig.output {
            ty::FnConverging(ret_ty) => {
                let return_type_id = self.get_unique_type_id_of_type(cx, ret_ty);
                let return_type_id = self.get_unique_type_id_as_string(return_type_id);
                unique_type_id.push_str(&return_type_id[..]);
            }
            ty::FnDiverging => {
                unique_type_id.push_str("!");
            }
        }
    }

    // Get the UniqueTypeId for an enum variant. Enum variants are not really
    // types of their own, so they need special handling. We still need a
    // UniqueTypeId for them, since to debuginfo they *are* real types.
    fn get_unique_type_id_of_enum_variant<'a>(&mut self,
                                              cx: &CrateContext<'a, 'tcx>,
                                              enum_type: Ty<'tcx>,
                                              variant_name: &str)
                                              -> UniqueTypeId {
        let enum_type_id = self.get_unique_type_id_of_type(cx, enum_type);
        let enum_variant_type_id = format!("{}::{}",
                                           &self.get_unique_type_id_as_string(enum_type_id),
                                           variant_name);
        let interner_key = self.unique_id_interner.intern(Rc::new(enum_variant_type_id));
        UniqueTypeId(interner_key)
    }
}


/// A context object for maintaining all state needed by the debuginfo module.
pub struct CrateDebugContext<'tcx> {
    llcontext: ContextRef,
    builder: DIBuilderRef,
    current_debug_location: Cell<InternalDebugLocation>,
    created_files: RefCell<FnvHashMap<String, DIFile>>,
    created_enum_disr_types: RefCell<DefIdMap<DIType>>,

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
            current_debug_location: Cell::new(UnknownLocation),
            created_files: RefCell::new(FnvHashMap()),
            created_enum_disr_types: RefCell::new(DefIdMap()),
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
                                    llvm::LLVMRustDebugMetadataVersion);
    };
}

/// Creates debug information for the given global variable.
///
/// Adds the created metadata nodes directly to the crate's IR.
pub fn create_global_var_metadata(cx: &CrateContext,
                                  node_id: ast::NodeId,
                                  global: ValueRef) {
    if cx.dbg_cx().is_none() {
        return;
    }

    // Don't create debuginfo for globals inlined from other crates. The other
    // crate should already contain debuginfo for it. More importantly, the
    // global might not even exist in un-inlined form anywhere which would lead
    // to a linker errors.
    if cx.external_srcs().borrow().contains_key(&node_id) {
        return;
    }

    let var_item = cx.tcx().map.get(node_id);

    let (name, span) = match var_item {
        ast_map::NodeItem(item) => {
            match item.node {
                ast::ItemStatic(..) => (item.ident.name, item.span),
                ast::ItemConst(..) => (item.ident.name, item.span),
                _ => {
                    cx.sess()
                      .span_bug(item.span,
                                &format!("debuginfo::\
                                         create_global_var_metadata() -
                                         Captured var-id refers to \
                                         unexpected ast_item variant: {:?}",
                                        var_item))
                }
            }
        },
        _ => cx.sess().bug(&format!("debuginfo::create_global_var_metadata() \
                                    - Captured var-id refers to unexpected \
                                    ast_map variant: {:?}",
                                   var_item))
    };

    let (file_metadata, line_number) = if span != codemap::DUMMY_SP {
        let loc = span_start(cx, span);
        (file_metadata(cx, &loc.file.name), loc.line as c_uint)
    } else {
        (UNKNOWN_FILE_METADATA, UNKNOWN_LINE_NUMBER)
    };

    let is_local_to_unit = is_node_local_to_unit(cx, node_id);
    let variable_type = ty::node_id_to_type(cx.tcx(), node_id);
    let type_metadata = type_metadata(cx, variable_type, span);
    let namespace_node = namespace_for_item(cx, ast_util::local_def(node_id));
    let var_name = token::get_name(name).to_string();
    let linkage_name =
        namespace_node.mangled_name_of_contained_item(&var_name[..]);
    let var_scope = namespace_node.scope;

    let var_name = CString::new(var_name).unwrap();
    let linkage_name = CString::new(linkage_name).unwrap();
    unsafe {
        llvm::LLVMDIBuilderCreateStaticVariable(DIB(cx),
                                                var_scope,
                                                var_name.as_ptr(),
                                                linkage_name.as_ptr(),
                                                file_metadata,
                                                line_number,
                                                type_metadata,
                                                is_local_to_unit,
                                                global,
                                                ptr::null_mut());
    }
}

/// Creates debug information for the given local variable.
///
/// This function assumes that there's a datum for each pattern component of the
/// local in `bcx.fcx.lllocals`.
/// Adds the created metadata nodes directly to the crate's IR.
pub fn create_local_var_metadata(bcx: Block, local: &ast::Local) {
    if bcx.unreachable.get() ||
       fn_should_be_ignored(bcx.fcx) ||
       bcx.sess().opts.debuginfo != FullDebugInfo  {
        return;
    }

    let cx = bcx.ccx();
    let def_map = &cx.tcx().def_map;
    let locals = bcx.fcx.lllocals.borrow();

    pat_util::pat_bindings(def_map, &*local.pat, |_, node_id, span, var_ident| {
        let datum = match locals.get(&node_id) {
            Some(datum) => datum,
            None => {
                bcx.sess().span_bug(span,
                    &format!("no entry in lllocals table for {}",
                            node_id));
            }
        };

        if unsafe { llvm::LLVMIsAAllocaInst(datum.val) } == ptr::null_mut() {
            cx.sess().span_bug(span, "debuginfo::create_local_var_metadata() - \
                                      Referenced variable location is not an alloca!");
        }

        let scope_metadata = scope_metadata(bcx.fcx, node_id, span);

        declare_local(bcx,
                      var_ident.node.name,
                      datum.ty,
                      scope_metadata,
                      DirectVariable { alloca: datum.val },
                      LocalVariable,
                      span);
    })
}

/// Creates debug information for a variable captured in a closure.
///
/// Adds the created metadata nodes directly to the crate's IR.
pub fn create_captured_var_metadata<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                                                node_id: ast::NodeId,
                                                env_pointer: ValueRef,
                                                env_index: usize,
                                                captured_by_ref: bool,
                                                span: Span) {
    if bcx.unreachable.get() ||
       fn_should_be_ignored(bcx.fcx) ||
       bcx.sess().opts.debuginfo != FullDebugInfo {
        return;
    }

    let cx = bcx.ccx();

    let ast_item = cx.tcx().map.find(node_id);

    let variable_name = match ast_item {
        None => {
            cx.sess().span_bug(span, "debuginfo::create_captured_var_metadata: node not found");
        }
        Some(ast_map::NodeLocal(pat)) | Some(ast_map::NodeArg(pat)) => {
            match pat.node {
                ast::PatIdent(_, ref path1, _) => {
                    path1.node.name
                }
                _ => {
                    cx.sess()
                      .span_bug(span,
                                &format!(
                                "debuginfo::create_captured_var_metadata() - \
                                 Captured var-id refers to unexpected \
                                 ast_map variant: {:?}",
                                 ast_item));
                }
            }
        }
        _ => {
            cx.sess()
              .span_bug(span,
                        &format!("debuginfo::create_captured_var_metadata() - \
                                 Captured var-id refers to unexpected \
                                 ast_map variant: {:?}",
                                ast_item));
        }
    };

    let variable_type = common::node_id_type(bcx, node_id);
    let scope_metadata = bcx.fcx.debug_context.get_ref(cx, span).fn_metadata;

    // env_pointer is the alloca containing the pointer to the environment,
    // so it's type is **EnvironmentType. In order to find out the type of
    // the environment we have to "dereference" two times.
    let llvm_env_data_type = common::val_ty(env_pointer).element_type()
                                                        .element_type();
    let byte_offset_of_var_in_env = machine::llelement_offset(cx,
                                                              llvm_env_data_type,
                                                              env_index);

    let address_operations = unsafe {
        [llvm::LLVMDIBuilderCreateOpDeref(),
         llvm::LLVMDIBuilderCreateOpPlus(),
         byte_offset_of_var_in_env as i64,
         llvm::LLVMDIBuilderCreateOpDeref()]
    };

    let address_op_count = if captured_by_ref {
        address_operations.len()
    } else {
        address_operations.len() - 1
    };

    let variable_access = IndirectVariable {
        alloca: env_pointer,
        address_operations: &address_operations[..address_op_count]
    };

    declare_local(bcx,
                  variable_name,
                  variable_type,
                  scope_metadata,
                  variable_access,
                  CapturedVariable,
                  span);
}

/// Creates debug information for a local variable introduced in the head of a
/// match-statement arm.
///
/// Adds the created metadata nodes directly to the crate's IR.
pub fn create_match_binding_metadata<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                                                 variable_name: ast::Name,
                                                 binding: BindingInfo<'tcx>) {
    if bcx.unreachable.get() ||
       fn_should_be_ignored(bcx.fcx) ||
       bcx.sess().opts.debuginfo != FullDebugInfo {
        return;
    }

    let scope_metadata = scope_metadata(bcx.fcx, binding.id, binding.span);
    let aops = unsafe {
        [llvm::LLVMDIBuilderCreateOpDeref()]
    };
    // Regardless of the actual type (`T`) we're always passed the stack slot
    // (alloca) for the binding. For ByRef bindings that's a `T*` but for ByMove
    // bindings we actually have `T**`. So to get the actual variable we need to
    // dereference once more. For ByCopy we just use the stack slot we created
    // for the binding.
    let var_access = match binding.trmode {
        TrByCopy(llbinding) => DirectVariable {
            alloca: llbinding
        },
        TrByMove => IndirectVariable {
            alloca: binding.llmatch,
            address_operations: &aops
        },
        TrByRef => DirectVariable {
            alloca: binding.llmatch
        }
    };

    declare_local(bcx,
                  variable_name,
                  binding.ty,
                  scope_metadata,
                  var_access,
                  LocalVariable,
                  binding.span);
}

/// Creates debug information for the given function argument.
///
/// This function assumes that there's a datum for each pattern component of the
/// argument in `bcx.fcx.lllocals`.
/// Adds the created metadata nodes directly to the crate's IR.
pub fn create_argument_metadata(bcx: Block, arg: &ast::Arg) {
    if bcx.unreachable.get() ||
       fn_should_be_ignored(bcx.fcx) ||
       bcx.sess().opts.debuginfo != FullDebugInfo {
        return;
    }

    let def_map = &bcx.tcx().def_map;
    let scope_metadata = bcx
                         .fcx
                         .debug_context
                         .get_ref(bcx.ccx(), arg.pat.span)
                         .fn_metadata;
    let locals = bcx.fcx.lllocals.borrow();

    pat_util::pat_bindings(def_map, &*arg.pat, |_, node_id, span, var_ident| {
        let datum = match locals.get(&node_id) {
            Some(v) => v,
            None => {
                bcx.sess().span_bug(span,
                    &format!("no entry in lllocals table for {}",
                            node_id));
            }
        };

        if unsafe { llvm::LLVMIsAAllocaInst(datum.val) } == ptr::null_mut() {
            bcx.sess().span_bug(span, "debuginfo::create_argument_metadata() - \
                                       Referenced variable location is not an alloca!");
        }

        let argument_index = {
            let counter = &bcx
                          .fcx
                          .debug_context
                          .get_ref(bcx.ccx(), span)
                          .argument_counter;
            let argument_index = counter.get();
            counter.set(argument_index + 1);
            argument_index
        };

        declare_local(bcx,
                      var_ident.node.name,
                      datum.ty,
                      scope_metadata,
                      DirectVariable { alloca: datum.val },
                      ArgumentVariable(argument_index),
                      span);
    })
}

pub fn get_cleanup_debug_loc_for_ast_node<'a, 'tcx>(cx: &CrateContext<'a, 'tcx>,
                                                    node_id: ast::NodeId,
                                                    node_span: Span,
                                                    is_block: bool)
                                                 -> NodeIdAndSpan {
    // A debug location needs two things:
    // (1) A span (of which only the beginning will actually be used)
    // (2) An AST node-id which will be used to look up the lexical scope
    //     for the location in the functions scope-map
    //
    // This function will calculate the debug location for compiler-generated
    // cleanup calls that are executed when control-flow leaves the
    // scope identified by `node_id`.
    //
    // For everything but block-like things we can simply take id and span of
    // the given expression, meaning that from a debugger's view cleanup code is
    // executed at the same source location as the statement/expr itself.
    //
    // Blocks are a special case. Here we want the cleanup to be linked to the
    // closing curly brace of the block. The *scope* the cleanup is executed in
    // is up to debate: It could either still be *within* the block being
    // cleaned up, meaning that locals from the block are still visible in the
    // debugger.
    // Or it could be in the scope that the block is contained in, so any locals
    // from within the block are already considered out-of-scope and thus not
    // accessible in the debugger anymore.
    //
    // The current implementation opts for the second option: cleanup of a block
    // already happens in the parent scope of the block. The main reason for
    // this decision is that scoping becomes controlflow dependent when variable
    // shadowing is involved and it's impossible to decide statically which
    // scope is actually left when the cleanup code is executed.
    // In practice it shouldn't make much of a difference.

    let mut cleanup_span = node_span;

    if is_block {
        // Not all blocks actually have curly braces (e.g. simple closure
        // bodies), in which case we also just want to return the span of the
        // whole expression.
        let code_snippet = cx.sess().codemap().span_to_snippet(node_span);
        if let Ok(code_snippet) = code_snippet {
            let bytes = code_snippet.as_bytes();

            if !bytes.is_empty() && &bytes[bytes.len()-1..] == b"}" {
                cleanup_span = Span {
                    lo: node_span.hi - codemap::BytePos(1),
                    hi: node_span.hi,
                    expn_id: node_span.expn_id
                };
            }
        }
    }

    NodeIdAndSpan {
        id: node_id,
        span: cleanup_span
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
                set_source_location(fcx, node_id, span);
            }
            DebugLoc::None => {
                clear_source_location(fcx);
            }
        }
    }
}

pub trait ToDebugLoc {
    fn debug_loc(&self) -> DebugLoc;
}

impl ToDebugLoc for ast::Expr {
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

/// Sets the current debug location at the beginning of the span.
///
/// Maps to a call to llvm::LLVMSetCurrentDebugLocation(...). The node_id
/// parameter is used to reliably find the correct visibility scope for the code
/// position.
pub fn set_source_location(fcx: &FunctionContext,
                           node_id: ast::NodeId,
                           span: Span) {
    match fcx.debug_context {
        FunctionDebugContext::DebugInfoDisabled => return,
        FunctionDebugContext::FunctionWithoutDebugInfo => {
            set_debug_location(fcx.ccx, UnknownLocation);
            return;
        }
        FunctionDebugContext::RegularContext(box ref function_debug_context) => {
            if function_debug_context.source_location_override.get() {
                // Just ignore any attempts to set a new debug location while
                // the override is active.
                return;
            }

            let cx = fcx.ccx;

            debug!("set_source_location: {}", cx.sess().codemap().span_to_string(span));

            if function_debug_context.source_locations_enabled.get() {
                let loc = span_start(cx, span);
                let scope = scope_metadata(fcx, node_id, span);

                set_debug_location(cx, InternalDebugLocation::new(scope,
                                                                  loc.line,
                                                                  loc.col.to_usize()));
            } else {
                set_debug_location(cx, UnknownLocation);
            }
        }
    }
}

/// This function makes sure that all debug locations emitted while executing
/// `wrapped_function` are set to the given `debug_loc`.
pub fn with_source_location_override<F, R>(fcx: &FunctionContext,
                                           debug_loc: DebugLoc,
                                           wrapped_function: F) -> R
    where F: FnOnce() -> R
{
    match fcx.debug_context {
        FunctionDebugContext::DebugInfoDisabled => {
            wrapped_function()
        }
        FunctionDebugContext::FunctionWithoutDebugInfo => {
            set_debug_location(fcx.ccx, UnknownLocation);
            wrapped_function()
        }
        FunctionDebugContext::RegularContext(box ref function_debug_context) => {
            if function_debug_context.source_location_override.get() {
                wrapped_function()
            } else {
                debug_loc.apply(fcx);
                function_debug_context.source_location_override.set(true);
                let result = wrapped_function();
                function_debug_context.source_location_override.set(false);
                result
            }
        }
    }
}

/// Clears the current debug location.
///
/// Instructions generated hereafter won't be assigned a source location.
pub fn clear_source_location(fcx: &FunctionContext) {
    if fn_should_be_ignored(fcx) {
        return;
    }

    set_debug_location(fcx.ccx, UnknownLocation);
}

/// Enables emitting source locations for the given functions.
///
/// Since we don't want source locations to be emitted for the function prelude,
/// they are disabled when beginning to translate a new function. This functions
/// switches source location emitting on and must therefore be called before the
/// first real statement/expression of the function is translated.
pub fn start_emitting_source_locations(fcx: &FunctionContext) {
    match fcx.debug_context {
        FunctionDebugContext::RegularContext(box ref data) => {
            data.source_locations_enabled.set(true)
        },
        _ => { /* safe to ignore */ }
    }
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
    set_debug_location(cx, UnknownLocation);

    if fn_ast_id == ast::DUMMY_NODE_ID {
        // This is a function not linked to any source location, so don't
        // generate debuginfo for it.
        return FunctionDebugContext::FunctionWithoutDebugInfo;
    }

    let empty_generics = ast_util::empty_generics();

    let fnitem = cx.tcx().map.get(fn_ast_id);

    let (name, fn_decl, generics, top_level_block, span, has_path) = match fnitem {
        ast_map::NodeItem(ref item) => {
            if contains_nodebug_attribute(&item.attrs) {
                return FunctionDebugContext::FunctionWithoutDebugInfo;
            }

            match item.node {
                ast::ItemFn(ref fn_decl, _, _, ref generics, ref top_level_block) => {
                    (item.ident.name, fn_decl, generics, top_level_block, item.span, true)
                }
                _ => {
                    cx.sess().span_bug(item.span,
                        "create_function_debug_context: item bound to non-function");
                }
            }
        }
        ast_map::NodeImplItem(impl_item) => {
            match impl_item.node {
                ast::MethodImplItem(ref sig, ref body) => {
                    if contains_nodebug_attribute(&impl_item.attrs) {
                        return FunctionDebugContext::FunctionWithoutDebugInfo;
                    }

                    (impl_item.ident.name,
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
        ast_map::NodeExpr(ref expr) => {
            match expr.node {
                ast::ExprClosure(_, ref fn_decl, ref top_level_block) => {
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
        ast_map::NodeTraitItem(trait_item) => {
            match trait_item.node {
                ast::MethodTraitItem(ref sig, Some(ref body)) => {
                    if contains_nodebug_attribute(&trait_item.attrs) {
                        return FunctionDebugContext::FunctionWithoutDebugInfo;
                    }

                    (trait_item.ident.name,
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
        ast_map::NodeForeignItem(..) |
        ast_map::NodeVariant(..) |
        ast_map::NodeStructCtor(..) => {
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
                                                  &*fn_decl,
                                                  param_substs,
                                                  span);
        llvm::LLVMDIBuilderCreateSubroutineType(DIB(cx), file_metadata, fn_signature)
    };

    // Get_template_parameters() will append a `<...>` clause to the function
    // name if necessary.
    let mut function_name = String::from_str(&token::get_name(name));
    let template_parameters = get_template_parameters(cx,
                                                      generics,
                                                      param_substs,
                                                      file_metadata,
                                                      &mut function_name);

    // There is no ast_map::Path for ast::ExprClosure-type functions. For now,
    // just don't put them into a namespace. In the future this could be improved
    // somehow (storing a path in the ast_map, or construct a path using the
    // enclosing function).
    let (linkage_name, containing_scope) = if has_path {
        let namespace_node = namespace_for_item(cx, ast_util::local_def(fn_ast_id));
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

    let scope_map = create_scope_map(cx,
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
                                        fn_decl: &ast::FnDecl,
                                        param_substs: &Substs<'tcx>,
                                        error_reporting_span: Span) -> DIArray {
        if cx.sess().opts.debuginfo == LimitedDebugInfo {
            return create_DIArray(DIB(cx), &[]);
        }

        let mut signature = Vec::with_capacity(fn_decl.inputs.len() + 1);

        // Return type -- llvm::DIBuilder wants this at index 0
        assert_type_for_node_id(cx, fn_ast_id, error_reporting_span);
        let return_type = ty::node_id_to_type(cx.tcx(), fn_ast_id);
        let return_type = monomorphize::apply_param_substs(cx.tcx(),
                                                           param_substs,
                                                           &return_type);
        if ty::type_is_nil(return_type) {
            signature.push(ptr::null_mut())
        } else {
            signature.push(type_metadata(cx, return_type, codemap::DUMMY_SP));
        }

        // Arguments types
        for arg in &fn_decl.inputs {
            assert_type_for_node_id(cx, arg.pat.id, arg.pat.span);
            let arg_type = ty::node_id_to_type(cx.tcx(), arg.pat.id);
            let arg_type = monomorphize::apply_param_substs(cx.tcx(),
                                                            param_substs,
                                                            &arg_type);
            signature.push(type_metadata(cx, arg_type, codemap::DUMMY_SP));
        }

        return create_DIArray(DIB(cx), &signature[..]);
    }

    fn get_template_parameters<'a, 'tcx>(cx: &CrateContext<'a, 'tcx>,
                                         generics: &ast::Generics,
                                         param_substs: &Substs<'tcx>,
                                         file_metadata: DIFile,
                                         name_to_append_suffix_to: &mut String)
                                         -> DIArray
    {
        let self_type = param_substs.self_ty();
        let self_type = monomorphize::normalize_associated_type(cx.tcx(), &self_type);

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

                let name = token::get_name(special_idents::type_self.name);

                let name = CString::new(name.as_bytes()).unwrap();
                let param_metadata = unsafe {
                    llvm::LLVMDIBuilderCreateTemplateTypeParameter(
                        DIB(cx),
                        file_metadata,
                        name.as_ptr(),
                        actual_self_type_metadata,
                        ptr::null_mut(),
                        0,
                        0)
                };

                template_params.push(param_metadata);
            }
        }

        // Handle other generic parameters
        let actual_types = param_substs.types.get_slice(subst::FnSpace);
        for (index, &ast::TyParam{ ident, .. }) in generics.ty_params.iter().enumerate() {
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
                let ident = token::get_ident(ident);
                let name = CString::new(ident.as_bytes()).unwrap();
                let param_metadata = unsafe {
                    llvm::LLVMDIBuilderCreateTemplateTypeParameter(
                        DIB(cx),
                        file_metadata,
                        name.as_ptr(),
                        actual_type_metadata,
                        ptr::null_mut(),
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

// A description of some recursive type. It can either be already finished (as
// with FinalMetadata) or it is not yet finished, but contains all information
// needed to generate the missing parts of the description. See the
// documentation section on Recursive Types at the top of this file for more
// information.
pub enum RecursiveTypeDescription<'tcx> {
    UnfinishedMetadata {
        unfinished_type: Ty<'tcx>,
        unique_type_id: UniqueTypeId,
        metadata_stub: DICompositeType,
        llvm_type: Type,
        member_description_factory: MemberDescriptionFactory<'tcx>,
    },
    FinalMetadata(DICompositeType)
}

fn create_and_register_recursive_type_forward_declaration<'a, 'tcx>(
    cx: &CrateContext<'a, 'tcx>,
    unfinished_type: Ty<'tcx>,
    unique_type_id: UniqueTypeId,
    metadata_stub: DICompositeType,
    llvm_type: Type,
    member_description_factory: MemberDescriptionFactory<'tcx>)
 -> RecursiveTypeDescription<'tcx> {

    // Insert the stub into the TypeMap in order to allow for recursive references
    let mut type_map = debug_context(cx).type_map.borrow_mut();
    type_map.register_unique_id_with_metadata(cx, unique_type_id, metadata_stub);
    type_map.register_type_with_metadata(cx, unfinished_type, metadata_stub);

    UnfinishedMetadata {
        unfinished_type: unfinished_type,
        unique_type_id: unique_type_id,
        metadata_stub: metadata_stub,
        llvm_type: llvm_type,
        member_description_factory: member_description_factory,
    }
}

impl<'tcx> RecursiveTypeDescription<'tcx> {
    // Finishes up the description of the type in question (mostly by providing
    // descriptions of the fields of the given type) and returns the final type
    // metadata.
    fn finalize<'a>(&self, cx: &CrateContext<'a, 'tcx>) -> MetadataCreationResult {
        match *self {
            FinalMetadata(metadata) => MetadataCreationResult::new(metadata, false),
            UnfinishedMetadata {
                unfinished_type,
                unique_type_id,
                metadata_stub,
                llvm_type,
                ref member_description_factory,
                ..
            } => {
                // Make sure that we have a forward declaration of the type in
                // the TypeMap so that recursive references are possible. This
                // will always be the case if the RecursiveTypeDescription has
                // been properly created through the
                // create_and_register_recursive_type_forward_declaration()
                // function.
                {
                    let type_map = debug_context(cx).type_map.borrow();
                    if type_map.find_metadata_for_unique_id(unique_type_id).is_none() ||
                       type_map.find_metadata_for_type(unfinished_type).is_none() {
                        cx.sess().bug(&format!("Forward declaration of potentially recursive type \
                                              '{}' was not found in TypeMap!",
                                              ppaux::ty_to_string(cx.tcx(), unfinished_type))
                                      );
                    }
                }

                // ... then create the member descriptions ...
                let member_descriptions =
                    member_description_factory.create_member_descriptions(cx);

                // ... and attach them to the stub to complete it.
                set_members_of_composite_type(cx,
                                              metadata_stub,
                                              llvm_type,
                                              &member_descriptions[..]);
                return MetadataCreationResult::new(metadata_stub, true);
            }
        }
    }
}

#[derive(Copy, Clone, PartialEq)]
enum InternalDebugLocation {
    KnownLocation { scope: DIScope, line: usize, col: usize },
    UnknownLocation
}

impl InternalDebugLocation {
    fn new(scope: DIScope, line: usize, col: usize) -> InternalDebugLocation {
        KnownLocation {
            scope: scope,
            line: line,
            col: col,
        }
    }
}

fn set_debug_location(cx: &CrateContext, debug_location: InternalDebugLocation) {
    if debug_location == debug_context(cx).current_debug_location.get() {
        return;
    }

    let metadata_node;

    match debug_location {
        KnownLocation { scope, line, .. } => {
            // Always set the column to zero like Clang and GCC
            let col = UNKNOWN_COLUMN_NUMBER;
            debug!("setting debug location to {} {}", line, col);

            unsafe {
                metadata_node = llvm::LLVMDIBuilderCreateDebugLocation(
                    debug_context(cx).llcontext,
                    line as c_uint,
                    col as c_uint,
                    scope,
                    ptr::null_mut());
            }
        }
        UnknownLocation => {
            debug!("clearing debug location ");
            metadata_node = ptr::null_mut();
        }
    };

    unsafe {
        llvm::LLVMSetCurrentDebugLocation(cx.raw_builder(), metadata_node);
    }

    debug_context(cx).current_debug_location.set(debug_location);
}
