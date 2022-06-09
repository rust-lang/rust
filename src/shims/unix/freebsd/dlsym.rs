use rustc_middle::mir;

use crate::*;
use helpers::check_arg_count;

#[derive(Debug, Copy, Clone)]
#[allow(non_camel_case_types)]
pub enum Dlsym {
    getentropy,
}

impl Dlsym {
    // Returns an error for unsupported symbols, and None if this symbol
    // should become a NULL pointer (pretend it does not exist).
    pub fn from_str<'tcx>(name: &str) -> InterpResult<'tcx, Option<Dlsym>> {
        throw_unsup_format!("unsupported FreeBSD dlsym: {}", name)
    }
}

impl<'mir, 'tcx: 'mir> EvalContextExt<'mir, 'tcx> for crate::MiriEvalContext<'mir, 'tcx> {}
pub trait EvalContextExt<'mir, 'tcx: 'mir>: crate::MiriEvalContextExt<'mir, 'tcx> {
    fn call_dlsym(
        &mut self,
        dlsym: Dlsym,
        _args: &[OpTy<'tcx, Tag>],
        _dest: &PlaceTy<'tcx, Tag>,
        ret: Option<mir::BasicBlock>,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        let _ret = ret.expect("we don't support any diverging dlsym");
        assert!(this.tcx.sess.target.os == "freebsd");

        match dlsym {}
    }
}
