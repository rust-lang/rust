use rustc_middle::mir;
use rustc_target::spec::abi::Abi;

use crate::helpers::target_os_is_unix;
use crate::*;
use shims::unix::dlsym as unix;
use shims::windows::dlsym as windows;

#[derive(Debug, Copy, Clone)]
#[allow(non_camel_case_types)]
pub enum Dlsym {
    Posix(unix::Dlsym),
    Windows(windows::Dlsym),
}

impl Dlsym {
    // Returns an error for unsupported symbols, and None if this symbol
    // should become a NULL pointer (pretend it does not exist).
    pub fn from_str<'tcx>(name: &[u8], target_os: &str) -> InterpResult<'tcx, Option<Dlsym>> {
        let name = &*String::from_utf8_lossy(name);
        Ok(match target_os {
            target if target_os_is_unix(target) =>
                unix::Dlsym::from_str(name, target)?.map(Dlsym::Posix),
            "windows" => windows::Dlsym::from_str(name)?.map(Dlsym::Windows),
            os => bug!("dlsym not implemented for target_os {}", os),
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
        match dlsym {
            Dlsym::Posix(dlsym) =>
                unix::EvalContextExt::call_dlsym(this, dlsym, abi, args, dest, ret),
            Dlsym::Windows(dlsym) =>
                windows::EvalContextExt::call_dlsym(this, dlsym, abi, args, dest, ret),
        }
    }
}
