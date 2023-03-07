use rustc_middle::mir;

use crate::*;

#[derive(Debug, Copy, Clone)]
#[allow(non_camel_case_types)]
pub enum Dlsym {}

impl Dlsym {
    // Returns an error for unsupported symbols, and None if this symbol
    // should become a NULL pointer (pretend it does not exist).
    pub fn from_str<'tcx>(name: &str) -> InterpResult<'tcx, Option<Dlsym>> {
        throw_unsup_format!("unsupported FreeBSD dlsym: {}", name)
    }
}

impl<'mir, 'tcx: 'mir> EvalContextExt<'mir, 'tcx> for crate::MiriInterpCx<'mir, 'tcx> {}
pub trait EvalContextExt<'mir, 'tcx: 'mir>: crate::MiriInterpCxExt<'mir, 'tcx> {
    fn call_dlsym(
        &mut self,
        dlsym: Dlsym,
        _args: &[OpTy<'tcx, Provenance>],
        _dest: &PlaceTy<'tcx, Provenance>,
        ret: Option<mir::BasicBlock>,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        let _ret = ret.expect("we don't support any diverging dlsym");
        assert!(this.tcx.sess.target.os == "freebsd");

        match dlsym {}

        //trace!("{:?}", this.dump_place(**dest));
        //this.go_to_block(ret);
        //Ok(())
    }
}
