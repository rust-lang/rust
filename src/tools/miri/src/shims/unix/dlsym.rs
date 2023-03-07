use rustc_middle::mir;
use rustc_target::spec::abi::Abi;

use crate::*;
use shims::unix::android::dlsym as android;
use shims::unix::freebsd::dlsym as freebsd;
use shims::unix::linux::dlsym as linux;
use shims::unix::macos::dlsym as macos;

#[derive(Debug, Copy, Clone)]
pub enum Dlsym {
    Android(android::Dlsym),
    FreeBsd(freebsd::Dlsym),
    Linux(linux::Dlsym),
    MacOs(macos::Dlsym),
}

impl Dlsym {
    // Returns an error for unsupported symbols, and None if this symbol
    // should become a NULL pointer (pretend it does not exist).
    pub fn from_str<'tcx>(name: &str, target_os: &str) -> InterpResult<'tcx, Option<Dlsym>> {
        Ok(match target_os {
            "android" => android::Dlsym::from_str(name)?.map(Dlsym::Android),
            "freebsd" => freebsd::Dlsym::from_str(name)?.map(Dlsym::FreeBsd),
            "linux" => linux::Dlsym::from_str(name)?.map(Dlsym::Linux),
            "macos" => macos::Dlsym::from_str(name)?.map(Dlsym::MacOs),
            _ => panic!("unsupported Unix OS {target_os}"),
        })
    }
}

impl<'mir, 'tcx: 'mir> EvalContextExt<'mir, 'tcx> for crate::MiriInterpCx<'mir, 'tcx> {}
pub trait EvalContextExt<'mir, 'tcx: 'mir>: crate::MiriInterpCxExt<'mir, 'tcx> {
    fn call_dlsym(
        &mut self,
        dlsym: Dlsym,
        abi: Abi,
        args: &[OpTy<'tcx, Provenance>],
        dest: &PlaceTy<'tcx, Provenance>,
        ret: Option<mir::BasicBlock>,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();

        this.check_abi(abi, Abi::C { unwind: false })?;

        match dlsym {
            Dlsym::Android(dlsym) =>
                android::EvalContextExt::call_dlsym(this, dlsym, args, dest, ret),
            Dlsym::FreeBsd(dlsym) =>
                freebsd::EvalContextExt::call_dlsym(this, dlsym, args, dest, ret),
            Dlsym::Linux(dlsym) => linux::EvalContextExt::call_dlsym(this, dlsym, args, dest, ret),
            Dlsym::MacOs(dlsym) => macos::EvalContextExt::call_dlsym(this, dlsym, args, dest, ret),
        }
    }
}
