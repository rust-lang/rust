use rustc_middle::mir;

use crate::*;

#[derive(Debug, Copy, Clone)]
pub enum Dlsym {}

impl Dlsym {
    // Returns an error for unsupported symbols, and None if this symbol
    // should become a NULL pointer (pretend it does not exist).
    pub fn from_str(name: &str) -> InterpResult<'static, Option<Dlsym>> {
        Ok(match &*name {
            "__pthread_get_minstack" => None,
            "getrandom" => None, // std falls back to syscall(SYS_getrandom, ...) when this is NULL.
            "statx" => None,     // std falls back to syscall(SYS_statx, ...) when this is NULL.
            _ => throw_unsup_format!("unsupported Linux dlsym: {}", name),
        })
    }
}

impl<'mir, 'tcx: 'mir> EvalContextExt<'mir, 'tcx> for crate::MiriEvalContext<'mir, 'tcx> {}
pub trait EvalContextExt<'mir, 'tcx: 'mir>: crate::MiriEvalContextExt<'mir, 'tcx> {
    fn call_dlsym(
        &mut self,
        dlsym: Dlsym,
        _args: &[OpTy<'tcx, Tag>],
        ret: Option<(&PlaceTy<'tcx, Tag>, mir::BasicBlock)>,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        let (_dest, _ret) = ret.expect("we don't support any diverging dlsym");
        assert!(this.tcx.sess.target.os == "linux");

        match dlsym {}
    }
}
