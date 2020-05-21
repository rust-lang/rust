use rustc_middle::mir;

use crate::*;
use helpers::check_arg_count;

#[derive(Debug, Copy, Clone)]
pub enum Dlsym {
    GetEntropy,
}

impl Dlsym {
    // Returns an error for unsupported symbols, and None if this symbol
    // should become a NULL pointer (pretend it does not exist).
    pub fn from_str(name: &[u8], target_os: &str) -> InterpResult<'static, Option<Dlsym>> {
        use self::Dlsym::*;
        let name = String::from_utf8_lossy(name);
        Ok(match target_os {
            "linux" => match &*name {
                "__pthread_get_minstack" => None,
                _ => throw_unsup_format!("unsupported Linux dlsym: {}", name),
            }
            "macos" => match &*name {
                "getentropy" => Some(GetEntropy),
                _ => throw_unsup_format!("unsupported macOS dlsym: {}", name),
            }
            "windows" => match &*name {
                "SetThreadStackGuarantee" => None,
                "AcquireSRWLockExclusive" => None,
                "GetSystemTimePreciseAsFileTime" => None,
                _ => throw_unsup_format!("unsupported Windows dlsym: {}", name),
            }
            os => bug!("dlsym not implemented for target_os {}", os),
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
        use self::Dlsym::*;

        let this = self.eval_context_mut();
        let (dest, ret) = ret.expect("we don't support any diverging dlsym");

        match dlsym {
            GetEntropy => {
                let &[ptr, len] = check_arg_count(args)?;
                let ptr = this.read_scalar(ptr)?.not_undef()?;
                let len = this.read_scalar(len)?.to_machine_usize(this)?;
                this.gen_random(ptr, len)?;
                this.write_null(dest)?;
            }
        }

        this.dump_place(*dest);
        this.go_to_block(ret);
        Ok(())
    }
}
