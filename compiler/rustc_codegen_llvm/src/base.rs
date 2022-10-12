//! Codegen the MIR to the LLVM IR.
//!
//! Hopefully useful general knowledge about codegen:
//!
//! * There's no way to find out the [`Ty`] type of a [`Value`]. Doing so
//!   would be "trying to get the eggs out of an omelette" (credit:
//!   pcwalton). You can, instead, find out its [`llvm::Type`] by calling [`val_ty`],
//!   but one [`llvm::Type`] corresponds to many [`Ty`]s; for instance, `tup(int, int,
//!   int)` and `rec(x=int, y=int, z=int)` will have the same [`llvm::Type`].
//!
//! [`Ty`]: rustc_middle::ty::Ty
//! [`val_ty`]: crate::common::val_ty

use super::ModuleLlvm;

use crate::attributes;
use crate::builder::Builder;
use crate::context::CodegenCx;
use crate::llvm;
use crate::value::Value;

use cstr::cstr;

use rustc_codegen_ssa::base::maybe_create_entry_wrapper;
use rustc_codegen_ssa::mono_item::MonoItemExt;
use rustc_codegen_ssa::traits::*;
use rustc_codegen_ssa::{ModuleCodegen, ModuleKind};
use rustc_data_structures::small_c_str::SmallCStr;
use rustc_middle::dep_graph;
use rustc_middle::middle::codegen_fn_attrs::CodegenFnAttrs;
use rustc_middle::mir::mono::{Linkage, Visibility};
use rustc_middle::ty::TyCtxt;
use rustc_session::config::DebugInfo;
use rustc_span::symbol::Symbol;
use rustc_target::spec::SanitizerSet;

use std::time::Instant;

pub struct ValueIter<'ll> {
    cur: Option<&'ll Value>,
    step: unsafe extern "C" fn(&'ll Value) -> Option<&'ll Value>,
}

impl<'ll> Iterator for ValueIter<'ll> {
    type Item = &'ll Value;

    fn next(&mut self) -> Option<&'ll Value> {
        let old = self.cur;
        if let Some(old) = old {
            self.cur = unsafe { (self.step)(old) };
        }
        old
    }
}

pub fn iter_globals(llmod: &llvm::Module) -> ValueIter<'_> {
    unsafe { ValueIter { cur: llvm::LLVMGetFirstGlobal(llmod), step: llvm::LLVMGetNextGlobal } }
}

pub fn compile_codegen_unit(tcx: TyCtxt<'_>, cgu_name: Symbol) -> (ModuleCodegen<ModuleLlvm>, u64) {
    let start_time = Instant::now();

    let dep_node = tcx.codegen_unit(cgu_name).codegen_dep_node(tcx);
    let (module, _) = tcx.dep_graph.with_task(
        dep_node,
        tcx,
        cgu_name,
        module_codegen,
        Some(dep_graph::hash_result),
    );
    let time_to_codegen = start_time.elapsed();

    // We assume that the cost to run LLVM on a CGU is proportional to
    // the time we needed for codegenning it.
    let cost = time_to_codegen.as_nanos() as u64;

    fn module_codegen(tcx: TyCtxt<'_>, cgu_name: Symbol) -> ModuleCodegen<ModuleLlvm> {
        let cgu = tcx.codegen_unit(cgu_name);
        let _prof_timer =
            tcx.prof.generic_activity_with_arg_recorder("codegen_module", |recorder| {
                recorder.record_arg(cgu_name.to_string());
                recorder.record_arg(cgu.size_estimate().to_string());
            });
        // Instantiate monomorphizations without filling out definitions yet...
        let llvm_module = ModuleLlvm::new(tcx, cgu_name.as_str());
        {
            let cx = CodegenCx::new(tcx, cgu, &llvm_module);
            let mono_items = cx.codegen_unit.items_in_deterministic_order(cx.tcx);
            for &(mono_item, (linkage, visibility)) in &mono_items {
                mono_item.predefine::<Builder<'_, '_, '_>>(&cx, linkage, visibility);
            }

            // ... and now that we have everything pre-defined, fill out those definitions.
            for &(mono_item, _) in &mono_items {
                mono_item.define::<Builder<'_, '_, '_>>(&cx);
            }

            // If this codegen unit contains the main function, also create the
            // wrapper here
            if let Some(entry) = maybe_create_entry_wrapper::<Builder<'_, '_, '_>>(&cx) {
                let attrs = attributes::sanitize_attrs(&cx, SanitizerSet::empty());
                attributes::apply_to_llfn(entry, llvm::AttributePlace::Function, &attrs);
            }

            // Finalize code coverage by injecting the coverage map. Note, the coverage map will
            // also be added to the `llvm.compiler.used` variable, created next.
            if cx.sess().instrument_coverage() {
                cx.coverageinfo_finalize();
            }

            // Create the llvm.used and llvm.compiler.used variables.
            if !cx.used_statics.borrow().is_empty() {
                cx.create_used_variable_impl(cstr!("llvm.used"), &*cx.used_statics.borrow());
            }
            if !cx.compiler_used_statics.borrow().is_empty() {
                cx.create_used_variable_impl(
                    cstr!("llvm.compiler.used"),
                    &*cx.compiler_used_statics.borrow(),
                );
            }

            // Run replace-all-uses-with for statics that need it. This must
            // happen after the llvm.used variables are created.
            for &(old_g, new_g) in cx.statics_to_rauw().borrow().iter() {
                unsafe {
                    let bitcast = llvm::LLVMConstPointerCast(new_g, cx.val_ty(old_g));
                    llvm::LLVMReplaceAllUsesWith(old_g, bitcast);
                    llvm::LLVMDeleteGlobal(old_g);
                }
            }

            // Finalize debuginfo
            if cx.sess().opts.debuginfo != DebugInfo::None {
                cx.debuginfo_finalize();
            }
        }

        ModuleCodegen {
            name: cgu_name.to_string(),
            module_llvm: llvm_module,
            kind: ModuleKind::Regular,
        }
    }

    (module, cost)
}

pub fn set_link_section(llval: &Value, attrs: &CodegenFnAttrs) {
    let Some(sect) = attrs.link_section else { return };
    unsafe {
        let buf = SmallCStr::new(sect.as_str());
        llvm::LLVMSetSection(llval, buf.as_ptr());
    }
}

pub fn linkage_to_llvm(linkage: Linkage) -> llvm::Linkage {
    match linkage {
        Linkage::External => llvm::Linkage::ExternalLinkage,
        Linkage::AvailableExternally => llvm::Linkage::AvailableExternallyLinkage,
        Linkage::LinkOnceAny => llvm::Linkage::LinkOnceAnyLinkage,
        Linkage::LinkOnceODR => llvm::Linkage::LinkOnceODRLinkage,
        Linkage::WeakAny => llvm::Linkage::WeakAnyLinkage,
        Linkage::WeakODR => llvm::Linkage::WeakODRLinkage,
        Linkage::Appending => llvm::Linkage::AppendingLinkage,
        Linkage::Internal => llvm::Linkage::InternalLinkage,
        Linkage::Private => llvm::Linkage::PrivateLinkage,
        Linkage::ExternalWeak => llvm::Linkage::ExternalWeakLinkage,
        Linkage::Common => llvm::Linkage::CommonLinkage,
    }
}

pub fn visibility_to_llvm(linkage: Visibility) -> llvm::Visibility {
    match linkage {
        Visibility::Default => llvm::Visibility::Default,
        Visibility::Hidden => llvm::Visibility::Hidden,
        Visibility::Protected => llvm::Visibility::Protected,
    }
}
