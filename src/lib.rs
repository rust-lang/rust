#![feature(rustc_private, decl_macro, type_alias_impl_trait, associated_type_bounds, never_type)]
#![allow(intra_doc_link_resolution_failure)]

extern crate flate2;
extern crate libc;
extern crate tempfile;
extern crate rustc;
extern crate rustc_codegen_ssa;
extern crate rustc_codegen_utils;
extern crate rustc_data_structures;
extern crate rustc_driver;
extern crate rustc_fs_util;
extern crate rustc_hir;
extern crate rustc_incremental;
extern crate rustc_index;
extern crate rustc_mir;
extern crate rustc_session;
extern crate rustc_span;
extern crate rustc_target;
extern crate rustc_ast;

use std::any::Any;

use rustc::dep_graph::{DepGraph, WorkProduct, WorkProductId};
use rustc::middle::cstore::{EncodedMetadata, MetadataLoader};
use rustc::session::config::OutputFilenames;
use rustc::ty::query::Providers;
use rustc::util::common::ErrorReported;
use rustc_codegen_utils::codegen_backend::CodegenBackend;

use cranelift_codegen::settings;

use crate::constant::ConstantCx;
use crate::prelude::*;

mod abi;
mod allocator;
mod analyze;
mod archive;
mod atomic_shim;
mod base;
mod backend;
mod cast;
mod codegen_i128;
mod common;
mod constant;
mod debuginfo;
mod discriminant;
mod driver;
mod intrinsics;
mod linkage;
mod main_shim;
mod metadata;
mod num;
mod optimize;
mod pointer;
mod pretty_clif;
mod target_features_whitelist;
mod trap;
mod unimpl;
mod unsize;
mod value_and_place;
mod vtable;

mod prelude {
    pub use std::any::Any;
    pub use std::collections::{HashMap, HashSet};
    pub use std::convert::{TryFrom, TryInto};

    pub use rustc_ast::ast::{FloatTy, IntTy, UintTy};
    pub use rustc_span::{Pos, Span};

    pub use rustc::bug;
    pub use rustc_hir::def_id::{CrateNum, DefId, LOCAL_CRATE};
    pub use rustc::mir::{self, interpret::AllocId, mono::MonoItem, *};
    pub use rustc::session::{
        config::{CrateType, Lto},
        Session,
    };
    pub use rustc::ty::layout::{self, Abi, LayoutOf, Scalar, Size, TyLayout, VariantIdx};
    pub use rustc::ty::{
        self, FnSig, Instance, InstanceDef, ParamEnv, PolyFnSig, Ty, TyCtxt, TypeAndMut,
        TypeFoldable,
    };

    pub use rustc_data_structures::{
        fx::{FxHashMap, FxHashSet},
        sync::Lrc,
    };

    pub use rustc_index::vec::Idx;

    pub use rustc_codegen_ssa::mir::operand::{OperandRef, OperandValue};
    pub use rustc_codegen_ssa::traits::*;
    pub use rustc_codegen_ssa::{CodegenResults, CompiledModule, ModuleKind};

    pub use cranelift_codegen::Context;
    pub use cranelift_codegen::entity::EntitySet;
    pub use cranelift_codegen::ir::{AbiParam, Block, ExternalName, FuncRef, Inst, InstBuilder, MemFlags, Signature, SourceLoc, StackSlot, StackSlotData, StackSlotKind, TrapCode, Type, Value};
    pub use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
    pub use cranelift_codegen::ir::function::Function;
    pub use cranelift_codegen::ir::immediates::{Ieee32, Ieee64};
    pub use cranelift_codegen::ir::types;
    pub use cranelift_codegen::isa::{self, CallConv};
    pub use cranelift_codegen::settings::{self, Configurable};
    pub use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
    pub use cranelift_module::{
        self, Backend, DataContext, DataId, FuncId, FuncOrDataId, Linkage, Module,
    };

    pub use crate::abi::*;
    pub use crate::base::{trans_operand, trans_place};
    pub use crate::cast::*;
    pub use crate::common::*;
    pub use crate::debuginfo::{DebugContext, FunctionDebugContext};
    pub use crate::pointer::Pointer;
    pub use crate::trap::*;
    pub use crate::unimpl::unimpl;
    pub use crate::value_and_place::{CPlace, CPlaceInner, CValue};
    pub use crate::CodegenCx;

    pub struct PrintOnPanic<F: Fn() -> String>(pub F);
    impl<F: Fn() -> String> Drop for PrintOnPanic<F> {
        fn drop(&mut self) {
            if ::std::thread::panicking() {
                println!("{}", (self.0)());
            }
        }
    }
}

pub struct CodegenCx<'clif, 'tcx, B: Backend + 'static> {
    tcx: TyCtxt<'tcx>,
    module: &'clif mut Module<B>,
    constants_cx: ConstantCx,
    cached_context: Context,
    vtables: HashMap<(Ty<'tcx>, Option<ty::PolyExistentialTraitRef<'tcx>>), DataId>,
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
            constants_cx: ConstantCx::default(),
            cached_context: Context::new(),
            vtables: HashMap::new(),
            debug_context,
        }
    }

    fn finalize(self) {
        self.constants_cx.finalize(self.tcx, self.module);
    }
}

struct CraneliftCodegenBackend;

impl CodegenBackend for CraneliftCodegenBackend {
    fn init(&self, sess: &Session) {
        if sess.lto() != rustc_session::config::Lto::No {
            sess.warn("LTO is not supported. You may get a linker error.");
        }
    }

    fn metadata_loader(&self) -> Box<dyn MetadataLoader + Sync> {
        Box::new(crate::metadata::CraneliftMetadataLoader)
    }

    fn provide(&self, providers: &mut Providers) {
        providers.target_features_whitelist = |tcx, cnum| {
            assert_eq!(cnum, LOCAL_CRATE);
            if tcx.sess.opts.actually_rustdoc {
                // rustdoc needs to be able to document functions that use all the features, so
                // whitelist them all
                tcx.arena.alloc(
                    target_features_whitelist::all_known_features()
                        .map(|(a, b)| (a.to_string(), b))
                        .collect(),
                )
            } else {
                tcx.arena.alloc(
                    target_features_whitelist::target_feature_whitelist(tcx.sess)
                        .iter()
                        .map(|&(a, b)| (a.to_string(), b))
                        .collect(),
                )
            }
        };
    }
    fn provide_extern(&self, _providers: &mut Providers) {}

    fn codegen_crate<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        metadata: EncodedMetadata,
        need_metadata_module: bool,
    ) -> Box<dyn Any> {
        rustc_codegen_utils::check_for_rustc_errors_attr(tcx);

        let res = driver::codegen_crate(tcx, metadata, need_metadata_module);

        rustc_incremental::assert_module_sources::assert_module_sources(tcx);
        rustc_codegen_utils::symbol_names_test::report_symbol_names(tcx);

        res
    }

    fn join_codegen(
        &self,
        ongoing_codegen: Box<dyn Any>,
        sess: &Session,
        dep_graph: &DepGraph,
    ) -> Result<Box<dyn Any>, ErrorReported> {
        let (codegen_results, work_products) = *ongoing_codegen.downcast::<(CodegenResults, FxHashMap<WorkProductId, WorkProduct>)>().unwrap();

        sess.time("serialize_work_products", move || {
            rustc_incremental::save_work_product_index(sess, &dep_graph, work_products)
        });

        Ok(Box::new(codegen_results))
    }

    fn link(
        &self,
        sess: &Session,
        res: Box<dyn Any>,
        outputs: &OutputFilenames,
    ) -> Result<(), ErrorReported> {
        use rustc_codegen_ssa::back::link::link_binary;

        let codegen_results = *res
            .downcast::<CodegenResults>()
            .expect("Expected CraneliftCodegenBackend's CodegenResult, found Box<Any>");

        let _timer = sess.prof.generic_activity("link_crate");

        sess.time("linking", || {
            let target_cpu = crate::target_triple(sess).to_string();
            link_binary::<crate::archive::ArArchiveBuilder<'_>>(
                sess,
                &codegen_results,
                outputs,
                &codegen_results.crate_name.as_str(),
                &target_cpu,
            );
        });

        rustc_incremental::finalize_session_directory(sess, codegen_results.crate_hash);

        Ok(())
    }
}

fn target_triple(sess: &Session) -> target_lexicon::Triple {
    sess.target.target.llvm_target.parse().unwrap()
}

fn build_isa(sess: &Session, enable_pic: bool) -> Box<dyn isa::TargetIsa + 'static> {
    use target_lexicon::BinaryFormat;

    let target_triple = crate::target_triple(sess);

    let mut flags_builder = settings::builder();
    if enable_pic {
        flags_builder.enable("is_pic").unwrap();
    } else {
        flags_builder.set("is_pic", "false").unwrap();
    }
    flags_builder.set("enable_probestack", "false").unwrap(); // __cranelift_probestack is not provided
    flags_builder
        .set(
            "enable_verifier",
            if cfg!(debug_assertions) {
                "true"
            } else {
                "false"
            },
        )
        .unwrap();

    let tls_model = match target_triple.binary_format {
        BinaryFormat::Elf => "elf_gd",
        BinaryFormat::Macho => "macho",
        BinaryFormat::Coff => "coff",
        _ => "none",
    };
    flags_builder.set("tls_model", tls_model).unwrap();

    // FIXME(CraneStation/cranelift#732) fix LICM in presence of jump tables
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

    let flags = settings::Flags::new(flags_builder);
    cranelift_codegen::isa::lookup(target_triple)
        .unwrap()
        .finish(flags)
}

/// This is the entrypoint for a hot plugged rustc_codegen_cranelift
#[no_mangle]
pub fn __rustc_codegen_backend() -> Box<dyn CodegenBackend> {
    Box::new(CraneliftCodegenBackend)
}
