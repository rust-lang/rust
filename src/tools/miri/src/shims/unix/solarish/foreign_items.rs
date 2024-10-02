use rustc_span::Symbol;
use rustc_target::spec::abi::Abi;

use crate::shims::unix::*;
use crate::*;

pub fn is_dyn_sym(name: &str) -> bool {
    matches!(name, "pthread_setname_np")
}

impl<'tcx> EvalContextExt<'tcx> for crate::MiriInterpCx<'tcx> {}
pub trait EvalContextExt<'tcx>: crate::MiriInterpCxExt<'tcx> {
    fn emulate_foreign_item_inner(
        &mut self,
        link_name: Symbol,
        abi: Abi,
        args: &[OpTy<'tcx>],
        dest: &MPlaceTy<'tcx>,
    ) -> InterpResult<'tcx, EmulateItemResult> {
        let this = self.eval_context_mut();
        match link_name.as_str() {
            // Threading
            "pthread_setname_np" => {
                let [thread, name] =
                    this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;
                // THREAD_NAME_MAX allows a thread name of 31+1 length
                // https://github.com/illumos/illumos-gate/blob/7671517e13b8123748eda4ef1ee165c6d9dba7fe/usr/src/uts/common/sys/thread.h#L613
                let max_len = 32;
                let res = this.pthread_setname_np(
                    this.read_scalar(thread)?,
                    this.read_scalar(name)?,
                    max_len,
                )?;
                this.write_scalar(res, dest)?;
            }
            "pthread_getname_np" => {
                let [thread, name, len] =
                    this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;
                let res = this.pthread_getname_np(
                    this.read_scalar(thread)?,
                    this.read_scalar(name)?,
                    this.read_scalar(len)?,
                )?;
                this.write_scalar(res, dest)?;
            }

            // Miscellaneous
            "___errno" => {
                let [] = this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;
                let errno_place = this.last_error_place()?;
                this.write_scalar(errno_place.to_ref(this).to_scalar(), dest)?;
            }

            "stack_getbounds" => {
                let [stack] = this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;
                let stack = this.deref_pointer_as(stack, this.libc_ty_layout("stack_t"))?;

                this.write_int_fields_named(
                    &[
                        ("ss_sp", this.machine.stack_addr.into()),
                        ("ss_size", this.machine.stack_size.into()),
                        // field set to 0 means not in an alternate signal stack
                        // https://docs.oracle.com/cd/E86824_01/html/E54766/stack-getbounds-3c.html
                        ("ss_flags", 0),
                    ],
                    &stack,
                )?;

                this.write_null(dest)?;
            }

            "pset_info" => {
                let [pset, tpe, cpus, list] =
                    this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;
                // We do not need to handle the current process cpu mask, available_parallelism
                // implementation pass null anyway. We only care for the number of
                // cpus.
                // https://docs.oracle.com/cd/E88353_01/html/E37841/pset-info-2.html

                let pset = this.read_scalar(pset)?.to_i32()?;
                let tpe = this.read_pointer(tpe)?;
                let list = this.read_pointer(list)?;

                let ps_myid = this.eval_libc_i32("PS_MYID");
                if ps_myid != pset {
                    throw_unsup_format!("pset_info is only supported with pset==PS_MYID");
                }

                if !this.ptr_is_null(tpe)? {
                    throw_unsup_format!("pset_info is only supported with type==NULL");
                }

                if !this.ptr_is_null(list)? {
                    throw_unsup_format!("pset_info is only supported with list==NULL");
                }

                let cpus = this.deref_pointer(cpus)?;
                this.write_scalar(Scalar::from_u32(this.machine.num_cpus), &cpus)?;
                this.write_null(dest)?;
            }

            _ => return interp_ok(EmulateItemResult::NotSupported),
        }
        interp_ok(EmulateItemResult::NeedsReturn)
    }
}
