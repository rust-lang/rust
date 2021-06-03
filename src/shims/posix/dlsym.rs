use rustc_middle::mir;
use rustc_target::spec::abi::Abi;

use crate::*;
use shims::posix::linux::dlsym as linux;
use shims::posix::macos::dlsym as macos;

#[derive(Debug, Copy, Clone)]
pub enum Dlsym {
    Linux(linux::Dlsym),
    MacOs(macos::Dlsym),
}

impl Dlsym {
    // Returns an error for unsupported symbols, and None if this symbol
    // should become a NULL pointer (pretend it does not exist).
    pub fn from_str(name: &str, target_os: &str) -> InterpResult<'static, Option<Dlsym>> {
        Ok(match target_os {
            "linux" => linux::Dlsym::from_str(name)?.map(Dlsym::Linux),
            "macos" => macos::Dlsym::from_str(name)?.map(Dlsym::MacOs),
            _ => unreachable!(),
        })
    }
}

impl<'mir, 'tcx: 'mir> EvalContextExt<'mir, 'tcx> for crate::MiriEvalContext<'mir, 'tcx> {}
pub trait EvalContextExt<'mir, 'tcx: 'mir>: crate::MiriEvalContextExt<'mir, 'tcx> {
    fn call_dlsym(
        &mut self,
        dlsym: Dlsym,
        abi: Abi,
        args: &[OpTy<'tcx, Tag>],
        ret: Option<(&PlaceTy<'tcx, Tag>, mir::BasicBlock)>,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();

        this.check_abi(abi, Abi::C { unwind: false })?;

        match dlsym {
            Dlsym::Linux(dlsym) => linux::EvalContextExt::call_dlsym(this, dlsym, args, ret),
            Dlsym::MacOs(dlsym) => macos::EvalContextExt::call_dlsym(this, dlsym, args, ret),
        }
    }
}
