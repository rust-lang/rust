use rustc_middle::mir;
use rustc_target::spec::abi::Abi;

use crate::*;
use shims::unix::linux::dlsym as linux;
use shims::unix::macos::dlsym as macos;
use shims::unix::freebsd::dlsym as freebsd;

#[derive(Debug, Copy, Clone)]
pub enum Dlsym {
    Linux(linux::Dlsym),
    MacOs(macos::Dlsym),
    FreeBSD(freebsd::Dlsym)
}

impl Dlsym {
    // Returns an error for unsupported symbols, and None if this symbol
    // should become a NULL pointer (pretend it does not exist).
    pub fn from_str<'tcx>(name: &str, target_os: &str) -> InterpResult<'tcx, Option<Dlsym>> {
        Ok(match target_os {
            "linux" => linux::Dlsym::from_str(name)?.map(Dlsym::Linux),
            "macos" => macos::Dlsym::from_str(name)?.map(Dlsym::MacOs),
            "freebsd" => freebsd::Dlsym::from_str(name)?.map(Dlsym::FreeBSD),
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
        dest: &PlaceTy<'tcx, Tag>,
        ret: Option<mir::BasicBlock>,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();

        this.check_abi(abi, Abi::C { unwind: false })?;

        match dlsym {
            Dlsym::Linux(dlsym) => linux::EvalContextExt::call_dlsym(this, dlsym, args, dest, ret),
            Dlsym::MacOs(dlsym) => macos::EvalContextExt::call_dlsym(this, dlsym, args, dest, ret),
            Dlsym::FreeBSD(dlsym) => freebsd::EvalContextExt::call_dlsym(this, dlsym, args, dest, ret)
        }
    }
}
