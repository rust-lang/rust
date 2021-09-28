use rustc_codegen_ssa::traits::PreDefineMethods;
use rustc_middle::middle::codegen_fn_attrs::CodegenFnAttrFlags;
use rustc_middle::mir::mono::{Linkage, Visibility};
use rustc_middle::ty::{self, Instance, TypeFoldable};
use rustc_middle::ty::layout::{FnAbiOf, LayoutOf};
use rustc_span::def_id::DefId;

use crate::base;
use crate::context::CodegenCx;
use crate::type_of::LayoutGccExt;

impl<'gcc, 'tcx> PreDefineMethods<'tcx> for CodegenCx<'gcc, 'tcx> {
    fn predefine_static(&self, def_id: DefId, _linkage: Linkage, _visibility: Visibility, symbol_name: &str) {
        let attrs = self.tcx.codegen_fn_attrs(def_id);
        let instance = Instance::mono(self.tcx, def_id);
        let ty = instance.ty(self.tcx, ty::ParamEnv::reveal_all());
        let gcc_type = self.layout_of(ty).gcc_type(self, true);

        let is_tls = attrs.flags.contains(CodegenFnAttrFlags::THREAD_LOCAL);
        let global = self.define_global(symbol_name, gcc_type, is_tls, attrs.link_section);

        // TODO(antoyo): set linkage and visibility.
        self.instances.borrow_mut().insert(instance, global);
    }

    fn predefine_fn(&self, instance: Instance<'tcx>, linkage: Linkage, _visibility: Visibility, symbol_name: &str) {
        assert!(!instance.substs.needs_infer());

        let fn_abi = self.fn_abi_of_instance(instance, ty::List::empty());
        self.linkage.set(base::linkage_to_gcc(linkage));
        let _decl = self.declare_fn(symbol_name, &fn_abi);
        //let attrs = self.tcx.codegen_fn_attrs(instance.def_id());

        // TODO(antoyo): call set_link_section() to allow initializing argc/argv.
        // TODO(antoyo): set unique comdat.
        // TODO(antoyo): use inline attribute from there in linkage.set() above.
    }
}
