#![feature(rustc_private, never_type, decl_macro)]
#![allow(intra_doc_link_resolution_failure)]

extern crate flate2;
extern crate rustc;
extern crate rustc_allocator;
extern crate rustc_codegen_ssa;
extern crate rustc_codegen_utils;
extern crate rustc_data_structures;
extern crate rustc_driver;
extern crate rustc_fs_util;
extern crate rustc_incremental;
extern crate rustc_mir;
extern crate rustc_target;
extern crate syntax;

use std::any::Any;
use std::sync::mpsc;

use rustc::dep_graph::DepGraph;
use rustc::middle::cstore::{EncodedMetadata, MetadataLoader};
use rustc::session::config::OutputFilenames;
use rustc::ty::query::Providers;
use rustc::util::common::ErrorReported;
use rustc_codegen_utils::codegen_backend::CodegenBackend;

use cranelift::codegen::settings;

use crate::constant::ConstantCx;
use crate::prelude::*;

mod abi;
mod allocator;
mod analyze;
mod archive;
mod base;
mod common;
mod constant;
mod debuginfo;
mod driver;
mod intrinsics;
mod linkage;
mod main_shim;
mod metadata;
mod pretty_clif;
mod trap;
mod unimpl;
mod unsize;
mod value_and_place;
mod vtable;

mod prelude {
    pub use std::any::Any;
    pub use std::collections::{HashMap, HashSet};
    pub use std::convert::TryInto;

    pub use syntax::ast::{FloatTy, IntTy, UintTy};
    pub use syntax::source_map::{Pos, Span, DUMMY_SP};

    pub use rustc::bug;
    pub use rustc::hir::def_id::{CrateNum, DefId, LOCAL_CRATE};
    pub use rustc::mir::{self, interpret::AllocId, mono::MonoItem, *};
    pub use rustc::session::{
        config::{CrateType, Lto},
        Session,
    };
    pub use rustc::ty::layout::{self, Abi, LayoutOf, Scalar, Size, TyLayout, VariantIdx};
    pub use rustc::ty::{
        self, FnSig, Instance, InstanceDef, ParamEnv, PolyFnSig, Ty, TyCtxt,
        TypeAndMut, TypeFoldable,
    };
    pub use rustc_data_structures::{
        fx::{FxHashMap, FxHashSet},
        indexed_vec::Idx,
        sync::Lrc,
    };
    pub use rustc_mir::monomorphize::collector;

    pub use rustc_codegen_ssa::mir::operand::{OperandRef, OperandValue};
    pub use rustc_codegen_ssa::traits::*;
    pub use rustc_codegen_ssa::{CodegenResults, CompiledModule, ModuleKind};

    pub use cranelift::codegen::ir::{
        condcodes::IntCC, function::Function, ExternalName, FuncRef, Inst, SourceLoc, StackSlot,
    };
    pub use cranelift::codegen::isa::CallConv;
    pub use cranelift::codegen::Context;
    pub use cranelift::prelude::*;
    pub use cranelift_module::{
        self, Backend, DataContext, DataId, FuncId, FuncOrDataId, Linkage, Module,
    };

    pub use crate::abi::*;
    pub use crate::base::{trans_operand, trans_place};
    pub use crate::common::*;
    pub use crate::debuginfo::{DebugContext, FunctionDebugContext};
    pub use crate::trap::*;
    pub use crate::unimpl::{unimpl, with_unimpl_span};
    pub use crate::value_and_place::{CValue, CPlace};
    pub use crate::{Caches, CodegenCx};

    pub struct PrintOnPanic<F: Fn() -> String>(pub F);
    impl<F: Fn() -> String> Drop for PrintOnPanic<F> {
        fn drop(&mut self) {
            if ::std::thread::panicking() {
                println!("{}", (self.0)());
            }
        }
    }
}

pub struct Caches<'tcx> {
    pub context: Context,
    pub vtables: HashMap<(Ty<'tcx>, Option<ty::PolyExistentialTraitRef<'tcx>>), DataId>,
}

impl Default for Caches<'_> {
    fn default() -> Self {
        Caches {
            context: Context::new(),
            vtables: HashMap::new(),
        }
    }
}

pub struct CodegenCx<'clif, 'tcx, B: Backend + 'static> {
    tcx: TyCtxt<'tcx>,
    module: &'clif mut Module<B>,
    ccx: ConstantCx,
    caches: Caches<'tcx>,
    debug_context: Option<&'clif mut DebugContext<'tcx>>,
}

impl<'clif, 'tcx, B: Backend + 'static> CodegenCx<'clif, 'tcx, B> {
    fn new(
        tcx: TyCtxt<'tcx>,
        module: &'clif mut Module<B>,
        debug_context: Option<&'clif mut DebugContext<'tcx>>,
    ) -> Self {
        CodegenCx {
            tcx,
            module,
            ccx: ConstantCx::default(),
            caches: Caches::default(),
            debug_context,
        }
    }

    fn finalize(self) {
        self.ccx.finalize(self.tcx, self.module);
    }
}

struct CraneliftCodegenBackend;

impl CodegenBackend for CraneliftCodegenBackend {
    fn init(&self, _sess: &Session) {}

    fn metadata_loader(&self) -> Box<dyn MetadataLoader + Sync> {
        Box::new(crate::metadata::CraneliftMetadataLoader)
    }

    fn provide(&self, providers: &mut Providers) {
        rustc_codegen_utils::symbol_names::provide(providers);
        rustc_codegen_ssa::back::symbol_export::provide(providers);

        providers.target_features_whitelist = |tcx, _cnum| tcx.arena.alloc(FxHashMap::default());
    }
    fn provide_extern(&self, providers: &mut Providers) {
        rustc_codegen_ssa::back::symbol_export::provide_extern(providers);
    }

    fn codegen_crate<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        metadata: EncodedMetadata,
        need_metadata_module: bool,
        _rx: mpsc::Receiver<Box<dyn Any + Send>>,
    ) -> Box<dyn Any> {
        driver::codegen_crate(tcx, metadata, need_metadata_module)
    }

    fn join_codegen_and_link(
        &self,
        res: Box<dyn Any>,
        sess: &Session,
        _dep_graph: &DepGraph,
        outputs: &OutputFilenames,
    ) -> Result<(), ErrorReported> {
        use rustc_codegen_ssa::back::link::link_binary;

        let codegen_results = *res
            .downcast::<CodegenResults>()
            .expect("Expected CraneliftCodegenBackend's CodegenResult, found Box<Any>");

        let target_cpu = ::target_lexicon::HOST.to_string();
        link_binary::<crate::archive::ArArchiveBuilder<'_>>(
            sess,
            &codegen_results,
            outputs,
            &codegen_results.crate_name.as_str(),
            &target_cpu,
        );

        Ok(())
    }
}

fn target_triple(sess: &Session) -> target_lexicon::Triple {
    let mut target = &*sess.target.target.llvm_target;

    // FIXME add support for x86_64-apple-macosx10.7.0 to target-lexicon
    if target.starts_with("x86_64-apple-macosx") {
        target = "x86_64-apple-darwin";
    }

    target.parse().unwrap()
}

fn build_isa(sess: &Session) -> Box<dyn isa::TargetIsa + 'static> {
    let mut flags_builder = settings::builder();
    flags_builder.enable("is_pic").unwrap();
    flags_builder.set("probestack_enabled", "false").unwrap(); // ___cranelift_probestack is not provided
        flags_builder.set("enable_verifier", if cfg!(debug_assertions) {
        "true"
    } else {
        "false"
    }).unwrap();

    // FIXME enable again when https://github.com/CraneStation/cranelift/issues/664 is fixed
    /*
    use rustc::session::config::OptLevel;
    match sess.opts.optimize {
        OptLevel::No => {
            flags_builder.set("opt_level", "fastest").unwrap();
        }
        OptLevel::Less | OptLevel::Default => {}
        OptLevel::Aggressive => {
            flags_builder.set("opt_level", "best").unwrap();
        }
        OptLevel::Size | OptLevel::SizeMin => {
            sess.warn("Optimizing for size is not supported. Just ignoring the request");
        }
    }*/

    let target_triple = target_triple(sess);
    let flags = settings::Flags::new(flags_builder);
    cranelift::codegen::isa::lookup(target_triple)
        .unwrap()
        .finish(flags)
}

/// This is the entrypoint for a hot plugged rustc_codegen_cranelift
#[no_mangle]
pub fn __rustc_codegen_backend() -> Box<dyn CodegenBackend> {
    Box::new(CraneliftCodegenBackend)
}
