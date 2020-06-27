use rustc_middle::mir;

use crate::*;
use helpers::check_arg_count;

#[derive(Debug, Copy, Clone)]
pub enum Dlsym {
    AcquireSRWLockExclusive,
    AcquireSRWLockShared,
}

impl Dlsym {
    // Returns an error for unsupported symbols, and None if this symbol
    // should become a NULL pointer (pretend it does not exist).
    pub fn from_str(name: &str) -> InterpResult<'static, Option<Dlsym>> {
        Ok(match name {
            "AcquireSRWLockExclusive" => Some(Dlsym::AcquireSRWLockExclusive),
            "AcquireSRWLockShared" => Some(Dlsym::AcquireSRWLockShared),
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
        assert!(this.tcx.sess.target.target.target_os == "windows");

        match dlsym {
            Dlsym::AcquireSRWLockExclusive => {
                let &[ptr] = check_arg_count(args)?;
                let lock = this.deref_operand(ptr)?; // points to ptr-sized data
                throw_unsup_format!("AcquireSRWLockExclusive is not actually implemented");
            }
            Dlsym::AcquireSRWLockShared => {
                let &[ptr] = check_arg_count(args)?;
                let lock = this.deref_operand(ptr)?; // points to ptr-sized data
                throw_unsup_format!("AcquireSRWLockExclusive is not actually implemented");
            }
        }

        this.dump_place(*dest);
        this.go_to_block(ret);
        Ok(())
    }
}
