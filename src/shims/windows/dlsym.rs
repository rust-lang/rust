use rustc_middle::mir;
use rustc_target::spec::abi::Abi;

use crate::*;

#[derive(Debug, Copy, Clone)]
pub enum Dlsym {}

impl Dlsym {
    // Returns an error for unsupported symbols, and None if this symbol
    // should become a NULL pointer (pretend it does not exist).
    pub fn from_str(name: &str) -> InterpResult<'static, Option<Dlsym>> {
        Ok(match name {
            "GetSystemTimePreciseAsFileTime" => None,
            _ => throw_unsup_format!("unsupported Windows dlsym: {}", name),
        })
    }
}

impl<'mir, 'tcx: 'mir> EvalContextExt<'mir, 'tcx> for crate::MiriEvalContext<'mir, 'tcx> {}
pub trait EvalContextExt<'mir, 'tcx: 'mir>: crate::MiriEvalContextExt<'mir, 'tcx> {
    fn call_dlsym(
        &mut self,
        dlsym: Dlsym,
        abi: Abi,
        _args: &[OpTy<'tcx, Tag>],
        ret: Option<(&PlaceTy<'tcx, Tag>, mir::BasicBlock)>,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        let (_dest, _ret) = ret.expect("we don't support any diverging dlsym");
        assert!(this.tcx.sess.target.os == "windows");

        this.check_abi(abi, Abi::System { unwind: false })?;

        match dlsym {}
    }
}
