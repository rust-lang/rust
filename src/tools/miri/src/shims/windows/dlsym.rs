use rustc_middle::mir;
use rustc_target::spec::abi::Abi;

use log::trace;

use crate::helpers::check_arg_count;
use crate::shims::windows::handle::{EvalContextExt as _, Handle, PseudoHandle};
use crate::shims::windows::sync::EvalContextExt as _;
use crate::*;

#[derive(Debug, Copy, Clone)]
pub enum Dlsym {
    SetThreadDescription,
    WaitOnAddress,
    WakeByAddressSingle,
}

impl Dlsym {
    // Returns an error for unsupported symbols, and None if this symbol
    // should become a NULL pointer (pretend it does not exist).
    pub fn from_str<'tcx>(name: &str) -> InterpResult<'tcx, Option<Dlsym>> {
        Ok(match name {
            "GetSystemTimePreciseAsFileTime" => None,
            "SetThreadDescription" => Some(Dlsym::SetThreadDescription),
            "WaitOnAddress" => Some(Dlsym::WaitOnAddress),
            "WakeByAddressSingle" => Some(Dlsym::WakeByAddressSingle),
            _ => throw_unsup_format!("unsupported Windows dlsym: {}", name),
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
        let ret = ret.expect("we don't support any diverging dlsym");
        assert!(this.tcx.sess.target.os == "windows");

        this.check_abi(abi, Abi::System { unwind: false })?;

        match dlsym {
            Dlsym::SetThreadDescription => {
                let [handle, name] = check_arg_count(args)?;

                let handle = this.read_scalar(handle)?;

                let name = this.read_wide_str(this.read_pointer(name)?)?;

                let thread = match Handle::from_scalar(handle, this)? {
                    Some(Handle::Thread(thread)) => thread,
                    Some(Handle::Pseudo(PseudoHandle::CurrentThread)) => this.get_active_thread(),
                    _ => this.invalid_handle("SetThreadDescription")?,
                };

                this.set_thread_name_wide(thread, &name);

                this.write_null(dest)?;
            }
            Dlsym::WaitOnAddress => {
                let [ptr_op, compare_op, size_op, timeout_op] = check_arg_count(args)?;

                this.WaitOnAddress(ptr_op, compare_op, size_op, timeout_op, dest)?;
            }
            Dlsym::WakeByAddressSingle => {
                let [ptr_op] = check_arg_count(args)?;

                this.WakeByAddressSingle(ptr_op)?;
            }
        }

        trace!("{:?}", this.dump_place(**dest));
        this.go_to_block(ret);
        Ok(())
    }
}
