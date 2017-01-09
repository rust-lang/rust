// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Walks the crate looking for items/impl-items/trait-items that have
//! either a `rustc_symbol_name` or `rustc_item_path` attribute and
//! generates an error giving, respectively, the symbol name or
//! item-path. This is used for unit testing the code that generates
//! paths etc in all kinds of annoying scenarios.

use attributes;
use base;
use consts;
use context::{CrateContext, SharedCrateContext};
use common;
use declare;
use glue::DropGlueKind;
use llvm;
use monomorphize::{self, Instance};
use rustc::dep_graph::DepNode;
use rustc::hir;
use rustc::hir::def_id::DefId;
use rustc::ty::{self, Ty, TyCtxt, TypeFoldable};
use rustc::ty::subst::Substs;
use rustc_const_eval::fatal_const_eval_err;
use syntax::ast::{self, NodeId};
use syntax::attr;
use type_of;
use glue;
use abi::{Abi, FnType};
use back::symbol_names;
use std::fmt::Write;
use std::iter;

#[derive(PartialEq, Eq, Clone, Copy, Debug, Hash)]
pub enum TransItem<'tcx> {
    DropGlue(DropGlueKind<'tcx>),
    Fn(Instance<'tcx>),
    Static(NodeId)
}

/// Describes how a translation item will be instantiated in object files.
#[derive(PartialEq, Eq, Clone, Copy, Debug, Hash)]
pub enum InstantiationMode {
    /// There will be exactly one instance of the given TransItem. It will have
    /// external linkage so that it can be linked to from other codegen units.
    GloballyShared,

    /// Each codegen unit containing a reference to the given TransItem will
    /// have its own private copy of the function (with internal linkage).
    LocalCopy,
}

impl<'a, 'tcx> TransItem<'tcx> {

    pub fn define(&self, ccx: &CrateContext<'a, 'tcx>) {
        debug!("BEGIN IMPLEMENTING '{} ({})' in cgu {}",
                  self.to_string(ccx.tcx()),
                  self.to_raw_string(),
                  ccx.codegen_unit().name());

        // (*) This code executes in the context of a dep-node for the
        // entire CGU. In some cases, we introduce dep-nodes for
        // particular items that we are translating (these nodes will
        // have read edges coming into the CGU node). These smaller
        // nodes are not needed for correctness -- we always
        // invalidate an entire CGU at a time -- but they enable
        // finer-grained testing, since you can write tests that check
        // that the incoming edges to a particular fn are from a
        // particular set.

        match *self {
            TransItem::Static(node_id) => {
                let def_id = ccx.tcx().map.local_def_id(node_id);
                let _task = ccx.tcx().dep_graph.in_task(DepNode::TransCrateItem(def_id)); // (*)
                let item = ccx.tcx().map.expect_item(node_id);
                if let hir::ItemStatic(_, m, _) = item.node {
                    match consts::trans_static(&ccx, m, item.id, &item.attrs) {
                        Ok(_) => { /* Cool, everything's alright. */ },
                        Err(err) => {
                            // FIXME: shouldn't this be a `span_err`?
                            fatal_const_eval_err(
                                ccx.tcx(), &err, item.span, "static");
                        }
                    };
                } else {
                    span_bug!(item.span, "Mismatch between hir::Item type and TransItem type")
                }
            }
            TransItem::Fn(instance) => {
                let _task = ccx.tcx().dep_graph.in_task(
                    DepNode::TransCrateItem(instance.def)); // (*)

                base::trans_instance(&ccx, instance);
            }
            TransItem::DropGlue(dg) => {
                glue::implement_drop_glue(&ccx, dg);
            }
        }

        debug!("END IMPLEMENTING '{} ({})' in cgu {}",
               self.to_string(ccx.tcx()),
               self.to_raw_string(),
               ccx.codegen_unit().name());
    }

    pub fn predefine(&self,
                     ccx: &CrateContext<'a, 'tcx>,
                     linkage: llvm::Linkage) {
        debug!("BEGIN PREDEFINING '{} ({})' in cgu {}",
               self.to_string(ccx.tcx()),
               self.to_raw_string(),
               ccx.codegen_unit().name());

        let symbol_name = ccx.symbol_map()
                             .get_or_compute(ccx.shared(), *self);

        debug!("symbol {}", &symbol_name);

        match *self {
            TransItem::Static(node_id) => {
                TransItem::predefine_static(ccx, node_id, linkage, &symbol_name);
            }
            TransItem::Fn(instance) => {
                TransItem::predefine_fn(ccx, instance, linkage, &symbol_name);
            }
            TransItem::DropGlue(dg) => {
                TransItem::predefine_drop_glue(ccx, dg, linkage, &symbol_name);
            }
        }

        debug!("END PREDEFINING '{} ({})' in cgu {}",
               self.to_string(ccx.tcx()),
               self.to_raw_string(),
               ccx.codegen_unit().name());
    }

    fn predefine_static(ccx: &CrateContext<'a, 'tcx>,
                        node_id: ast::NodeId,
                        linkage: llvm::Linkage,
                        symbol_name: &str) {
        let def_id = ccx.tcx().map.local_def_id(node_id);
        let ty = ccx.tcx().item_type(def_id);
        let llty = type_of::type_of(ccx, ty);

        let g = declare::define_global(ccx, symbol_name, llty).unwrap_or_else(|| {
            ccx.sess().span_fatal(ccx.tcx().map.span(node_id),
                &format!("symbol `{}` is already defined", symbol_name))
        });

        unsafe { llvm::LLVMRustSetLinkage(g, linkage) };

        let instance = Instance::mono(ccx.shared(), def_id);
        ccx.instances().borrow_mut().insert(instance, g);
        ccx.statics().borrow_mut().insert(g, def_id);
    }

    fn predefine_fn(ccx: &CrateContext<'a, 'tcx>,
                    instance: Instance<'tcx>,
                    linkage: llvm::Linkage,
                    symbol_name: &str) {
        assert!(!instance.substs.needs_infer() &&
                !instance.substs.has_param_types());

        let item_ty = ccx.tcx().item_type(instance.def);
        let item_ty = ccx.tcx().erase_regions(&item_ty);
        let mono_ty = monomorphize::apply_param_substs(ccx.shared(), instance.substs, &item_ty);

        let attrs = ccx.tcx().get_attrs(instance.def);
        let lldecl = declare::declare_fn(ccx, symbol_name, mono_ty);
        unsafe { llvm::LLVMRustSetLinkage(lldecl, linkage) };
        base::set_link_section(ccx, lldecl, &attrs);
        if linkage == llvm::Linkage::LinkOnceODRLinkage ||
            linkage == llvm::Linkage::WeakODRLinkage {
            llvm::SetUniqueComdat(ccx.llmod(), lldecl);
        }

        if let ty::TyClosure(..) = mono_ty.sty {
            // set an inline hint for all closures
            attributes::inline(lldecl, attributes::InlineAttr::Hint);
        }

        attributes::from_fn_attrs(ccx, &attrs, lldecl);

        ccx.instances().borrow_mut().insert(instance, lldecl);
    }

    fn predefine_drop_glue(ccx: &CrateContext<'a, 'tcx>,
                           dg: glue::DropGlueKind<'tcx>,
                           linkage: llvm::Linkage,
                           symbol_name: &str) {
        let tcx = ccx.tcx();
        assert_eq!(dg.ty(), glue::get_drop_glue_type(ccx.shared(), dg.ty()));
        let t = dg.ty();

        let sig = tcx.mk_fn_sig(iter::once(tcx.mk_mut_ptr(t)), tcx.mk_nil(), false);

        debug!("predefine_drop_glue: sig={}", sig);

        let fn_ty = FnType::new(ccx, Abi::Rust, &sig, &[]);
        let llfnty = fn_ty.llvm_type(ccx);

        assert!(declare::get_defined_value(ccx, symbol_name).is_none());
        let llfn = declare::declare_cfn(ccx, symbol_name, llfnty);
        unsafe { llvm::LLVMRustSetLinkage(llfn, linkage) };
        if linkage == llvm::Linkage::LinkOnceODRLinkage ||
           linkage == llvm::Linkage::WeakODRLinkage {
            llvm::SetUniqueComdat(ccx.llmod(), llfn);
        }
        attributes::set_frame_pointer_elimination(ccx, llfn);
        ccx.drop_glues().borrow_mut().insert(dg, (llfn, fn_ty));
    }

    pub fn compute_symbol_name(&self,
                               scx: &SharedCrateContext<'a, 'tcx>) -> String {
        match *self {
            TransItem::Fn(instance) => instance.symbol_name(scx),
            TransItem::Static(node_id) => {
                let def_id = scx.tcx().map.local_def_id(node_id);
                Instance::mono(scx, def_id).symbol_name(scx)
            }
            TransItem::DropGlue(dg) => {
                let prefix = match dg {
                    DropGlueKind::Ty(_) => "drop",
                    DropGlueKind::TyContents(_) => "drop_contents",
                };
                symbol_names::exported_name_from_type_and_prefix(scx, dg.ty(), prefix)
            }
        }
    }

    pub fn is_from_extern_crate(&self) -> bool {
        match *self {
            TransItem::Fn(ref instance) => !instance.def.is_local(),
            TransItem::DropGlue(..) |
            TransItem::Static(..)   => false,
        }
    }

    pub fn instantiation_mode(&self,
                              tcx: TyCtxt<'a, 'tcx, 'tcx>)
                              -> InstantiationMode {
        match *self {
            TransItem::Fn(ref instance) => {
                if self.explicit_linkage(tcx).is_none() &&
                   (common::is_closure(tcx, instance.def) ||
                    attr::requests_inline(&tcx.get_attrs(instance.def)[..])) {
                    InstantiationMode::LocalCopy
                } else {
                    InstantiationMode::GloballyShared
                }
            }
            TransItem::DropGlue(..) => InstantiationMode::LocalCopy,
            TransItem::Static(..) => InstantiationMode::GloballyShared,
        }
    }

    pub fn is_generic_fn(&self) -> bool {
        match *self {
            TransItem::Fn(ref instance) => {
                instance.substs.types().next().is_some()
            }
            TransItem::DropGlue(..) |
            TransItem::Static(..)   => false,
        }
    }

    pub fn explicit_linkage(&self, tcx: TyCtxt<'a, 'tcx, 'tcx>) -> Option<llvm::Linkage> {
        let def_id = match *self {
            TransItem::Fn(ref instance) => instance.def,
            TransItem::Static(node_id) => tcx.map.local_def_id(node_id),
            TransItem::DropGlue(..) => return None,
        };

        let attributes = tcx.get_attrs(def_id);
        if let Some(name) = attr::first_attr_value_str_by_name(&attributes, "linkage") {
            if let Some(linkage) = base::llvm_linkage_by_name(&name.as_str()) {
                Some(linkage)
            } else {
                let span = tcx.map.span_if_local(def_id);
                if let Some(span) = span {
                    tcx.sess.span_fatal(span, "invalid linkage specified")
                } else {
                    tcx.sess.fatal(&format!("invalid linkage specified: {}", name))
                }
            }
        } else {
            None
        }
    }

    pub fn to_string(&self, tcx: TyCtxt<'a, 'tcx, 'tcx>) -> String {
        let hir_map = &tcx.map;

        return match *self {
            TransItem::DropGlue(dg) => {
                let mut s = String::with_capacity(32);
                match dg {
                    DropGlueKind::Ty(_) => s.push_str("drop-glue "),
                    DropGlueKind::TyContents(_) => s.push_str("drop-glue-contents "),
                };
                let printer = DefPathBasedNames::new(tcx, false, false);
                printer.push_type_name(dg.ty(), &mut s);
                s
            }
            TransItem::Fn(instance) => {
                to_string_internal(tcx, "fn ", instance)
            },
            TransItem::Static(node_id) => {
                let def_id = hir_map.local_def_id(node_id);
                let instance = Instance::new(def_id, tcx.intern_substs(&[]));
                to_string_internal(tcx, "static ", instance)
            },
        };

        fn to_string_internal<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                        prefix: &str,
                                        instance: Instance<'tcx>)
                                        -> String {
            let mut result = String::with_capacity(32);
            result.push_str(prefix);
            let printer = DefPathBasedNames::new(tcx, false, false);
            printer.push_instance_as_string(instance, &mut result);
            result
        }
    }

    pub fn to_raw_string(&self) -> String {
        match *self {
            TransItem::DropGlue(dg) => {
                let prefix = match dg {
                    DropGlueKind::Ty(_) => "Ty",
                    DropGlueKind::TyContents(_) => "TyContents",
                };
                format!("DropGlue({}: {})", prefix, dg.ty() as *const _ as usize)
            }
            TransItem::Fn(instance) => {
                format!("Fn({:?}, {})",
                         instance.def,
                         instance.substs.as_ptr() as usize)
            }
            TransItem::Static(id) => {
                format!("Static({:?})", id)
            }
        }
    }
}


//=-----------------------------------------------------------------------------
// TransItem String Keys
//=-----------------------------------------------------------------------------

// The code below allows for producing a unique string key for a trans item.
// These keys are used by the handwritten auto-tests, so they need to be
// predictable and human-readable.
//
// Note: A lot of this could looks very similar to what's already in the
//       ppaux module. It would be good to refactor things so we only have one
//       parameterizable implementation for printing types.

/// Same as `unique_type_name()` but with the result pushed onto the given
/// `output` parameter.
pub struct DefPathBasedNames<'a, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    omit_disambiguators: bool,
    omit_local_crate_name: bool,
}

impl<'a, 'tcx> DefPathBasedNames<'a, 'tcx> {
    pub fn new(tcx: TyCtxt<'a, 'tcx, 'tcx>,
               omit_disambiguators: bool,
               omit_local_crate_name: bool)
               -> Self {
        DefPathBasedNames {
            tcx: tcx,
            omit_disambiguators: omit_disambiguators,
            omit_local_crate_name: omit_local_crate_name,
        }
    }

    pub fn push_type_name(&self, t: Ty<'tcx>, output: &mut String) {
        match t.sty {
            ty::TyBool              => output.push_str("bool"),
            ty::TyChar              => output.push_str("char"),
            ty::TyStr               => output.push_str("str"),
            ty::TyNever             => output.push_str("!"),
            ty::TyInt(ast::IntTy::Is)    => output.push_str("isize"),
            ty::TyInt(ast::IntTy::I8)    => output.push_str("i8"),
            ty::TyInt(ast::IntTy::I16)   => output.push_str("i16"),
            ty::TyInt(ast::IntTy::I32)   => output.push_str("i32"),
            ty::TyInt(ast::IntTy::I64)   => output.push_str("i64"),
            ty::TyInt(ast::IntTy::I128)   => output.push_str("i128"),
            ty::TyUint(ast::UintTy::Us)   => output.push_str("usize"),
            ty::TyUint(ast::UintTy::U8)   => output.push_str("u8"),
            ty::TyUint(ast::UintTy::U16)  => output.push_str("u16"),
            ty::TyUint(ast::UintTy::U32)  => output.push_str("u32"),
            ty::TyUint(ast::UintTy::U64)  => output.push_str("u64"),
            ty::TyUint(ast::UintTy::U128)  => output.push_str("u128"),
            ty::TyFloat(ast::FloatTy::F32) => output.push_str("f32"),
            ty::TyFloat(ast::FloatTy::F64) => output.push_str("f64"),
            ty::TyAdt(adt_def, substs) => {
                self.push_def_path(adt_def.did, output);
                self.push_type_params(substs, iter::empty(), output);
            },
            ty::TyTuple(component_types) => {
                output.push('(');
                for &component_type in component_types {
                    self.push_type_name(component_type, output);
                    output.push_str(", ");
                }
                if !component_types.is_empty() {
                    output.pop();
                    output.pop();
                }
                output.push(')');
            },
            ty::TyBox(inner_type) => {
                output.push_str("Box<");
                self.push_type_name(inner_type, output);
                output.push('>');
            },
            ty::TyRawPtr(ty::TypeAndMut { ty: inner_type, mutbl } ) => {
                output.push('*');
                match mutbl {
                    hir::MutImmutable => output.push_str("const "),
                    hir::MutMutable => output.push_str("mut "),
                }

                self.push_type_name(inner_type, output);
            },
            ty::TyRef(_, ty::TypeAndMut { ty: inner_type, mutbl }) => {
                output.push('&');
                if mutbl == hir::MutMutable {
                    output.push_str("mut ");
                }

                self.push_type_name(inner_type, output);
            },
            ty::TyArray(inner_type, len) => {
                output.push('[');
                self.push_type_name(inner_type, output);
                write!(output, "; {}", len).unwrap();
                output.push(']');
            },
            ty::TySlice(inner_type) => {
                output.push('[');
                self.push_type_name(inner_type, output);
                output.push(']');
            },
            ty::TyDynamic(ref trait_data, ..) => {
                if let Some(principal) = trait_data.principal() {
                    self.push_def_path(principal.def_id(), output);
                    self.push_type_params(principal.skip_binder().substs,
                        trait_data.projection_bounds(),
                        output);
                }
            },
            ty::TyFnDef(.., &ty::BareFnTy{ unsafety, abi, ref sig } ) |
            ty::TyFnPtr(&ty::BareFnTy{ unsafety, abi, ref sig } ) => {
                if unsafety == hir::Unsafety::Unsafe {
                    output.push_str("unsafe ");
                }

                if abi != ::abi::Abi::Rust {
                    output.push_str("extern \"");
                    output.push_str(abi.name());
                    output.push_str("\" ");
                }

                output.push_str("fn(");

                let sig = self.tcx.erase_late_bound_regions_and_normalize(sig);

                if !sig.inputs().is_empty() {
                    for &parameter_type in sig.inputs() {
                        self.push_type_name(parameter_type, output);
                        output.push_str(", ");
                    }
                    output.pop();
                    output.pop();
                }

                if sig.variadic {
                    if !sig.inputs().is_empty() {
                        output.push_str(", ...");
                    } else {
                        output.push_str("...");
                    }
                }

                output.push(')');

                if !sig.output().is_nil() {
                    output.push_str(" -> ");
                    self.push_type_name(sig.output(), output);
                }
            },
            ty::TyClosure(def_id, ref closure_substs) => {
                self.push_def_path(def_id, output);
                let generics = self.tcx.item_generics(self.tcx.closure_base_def_id(def_id));
                let substs = closure_substs.substs.truncate_to(self.tcx, generics);
                self.push_type_params(substs, iter::empty(), output);
            }
            ty::TyError |
            ty::TyInfer(_) |
            ty::TyProjection(..) |
            ty::TyParam(_) |
            ty::TyAnon(..) => {
                bug!("DefPathBasedNames: Trying to create type name for \
                                         unexpected type: {:?}", t);
            }
        }
    }

    pub fn push_def_path(&self,
                         def_id: DefId,
                         output: &mut String) {
        let def_path = self.tcx.def_path(def_id);

        // some_crate::
        if !(self.omit_local_crate_name && def_id.is_local()) {
            output.push_str(&self.tcx.crate_name(def_path.krate).as_str());
            output.push_str("::");
        }

        // foo::bar::ItemName::
        for part in self.tcx.def_path(def_id).data {
            if self.omit_disambiguators {
                write!(output, "{}::", part.data.as_interned_str()).unwrap();
            } else {
                write!(output, "{}[{}]::",
                       part.data.as_interned_str(),
                       part.disambiguator).unwrap();
            }
        }

        // remove final "::"
        output.pop();
        output.pop();
    }

    fn push_type_params<I>(&self,
                            substs: &Substs<'tcx>,
                            projections: I,
                            output: &mut String)
        where I: Iterator<Item=ty::PolyExistentialProjection<'tcx>>
    {
        let mut projections = projections.peekable();
        if substs.types().next().is_none() && projections.peek().is_none() {
            return;
        }

        output.push('<');

        for type_parameter in substs.types() {
            self.push_type_name(type_parameter, output);
            output.push_str(", ");
        }

        for projection in projections {
            let projection = projection.skip_binder();
            let name = &projection.item_name.as_str();
            output.push_str(name);
            output.push_str("=");
            self.push_type_name(projection.ty, output);
            output.push_str(", ");
        }

        output.pop();
        output.pop();

        output.push('>');
    }

    pub fn push_instance_as_string(&self,
                                   instance: Instance<'tcx>,
                                   output: &mut String) {
        self.push_def_path(instance.def, output);
        self.push_type_params(instance.substs, iter::empty(), output);
    }
}
