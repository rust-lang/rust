use rustc_middle::mir;

use log::trace;

use crate::*;
use helpers::check_arg_count;
use shims::windows::sync::EvalContextExt as _;

#[derive(Debug, Copy, Clone)]
pub enum Dlsym {
    AcquireSRWLockExclusive,
    ReleaseSRWLockExclusive,
    TryAcquireSRWLockExclusive,
    AcquireSRWLockShared,
    ReleaseSRWLockShared,
    TryAcquireSRWLockShared,
}

impl Dlsym {
    // Returns an error for unsupported symbols, and None if this symbol
    // should become a NULL pointer (pretend it does not exist).
    pub fn from_str(name: &str) -> InterpResult<'static, Option<Dlsym>> {
        Ok(match name {
            "AcquireSRWLockExclusive" => Some(Dlsym::AcquireSRWLockExclusive),
            "ReleaseSRWLockExclusive" => Some(Dlsym::ReleaseSRWLockExclusive),
            "TryAcquireSRWLockExclusive" => Some(Dlsym::TryAcquireSRWLockExclusive),
            "AcquireSRWLockShared" => Some(Dlsym::AcquireSRWLockShared),
            "ReleaseSRWLockShared" => Some(Dlsym::ReleaseSRWLockShared),
            "TryAcquireSRWLockShared" => Some(Dlsym::TryAcquireSRWLockShared),
            "SetThreadStackGuarantee" => None,
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
        args: &[OpTy<'tcx, Tag>],
        ret: Option<(PlaceTy<'tcx, Tag>, mir::BasicBlock)>,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        let (dest, ret) = ret.expect("we don't support any diverging dlsym");
        assert!(this.tcx.sess.target.os == "windows");

        match dlsym {
            Dlsym::AcquireSRWLockExclusive => {
                let &[ptr] = check_arg_count(args)?;
                this.AcquireSRWLockExclusive(ptr)?;
            }
            Dlsym::ReleaseSRWLockExclusive => {
                let &[ptr] = check_arg_count(args)?;
                this.ReleaseSRWLockExclusive(ptr)?;
            }
            Dlsym::TryAcquireSRWLockExclusive => {
                let &[ptr] = check_arg_count(args)?;
                let ret = this.TryAcquireSRWLockExclusive(ptr)?;
                this.write_scalar(Scalar::from_u8(ret), dest)?;
            }
            Dlsym::AcquireSRWLockShared => {
                let &[ptr] = check_arg_count(args)?;
                this.AcquireSRWLockShared(ptr)?;
            }
            Dlsym::ReleaseSRWLockShared => {
                let &[ptr] = check_arg_count(args)?;
                this.ReleaseSRWLockShared(ptr)?;
            }
            Dlsym::TryAcquireSRWLockShared => {
                let &[ptr] = check_arg_count(args)?;
                let ret = this.TryAcquireSRWLockShared(ptr)?;
                this.write_scalar(Scalar::from_u8(ret), dest)?;
            }
        }

        trace!("{:?}", this.dump_place(*dest));
        this.go_to_block(ret);
        Ok(())
    }
}
