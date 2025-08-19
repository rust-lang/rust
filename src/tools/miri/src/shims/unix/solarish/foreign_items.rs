use rustc_abi::CanonAbi;
use rustc_middle::ty::Ty;
use rustc_span::Symbol;
use rustc_target::callconv::FnAbi;

use crate::shims::unix::foreign_items::EvalContextExt as _;
use crate::shims::unix::linux_like::epoll::EvalContextExt as _;
use crate::shims::unix::linux_like::eventfd::EvalContextExt as _;
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
        abi: &FnAbi<'tcx, Ty<'tcx>>,
        args: &[OpTy<'tcx>],
        dest: &MPlaceTy<'tcx>,
    ) -> InterpResult<'tcx, EmulateItemResult> {
        let this = self.eval_context_mut();
        match link_name.as_str() {
            // epoll, eventfd (NOT available on Solaris!)
            "epoll_create1" => {
                this.assert_target_os("illumos", "epoll_create1");
                let [flag] = this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;
                let result = this.epoll_create1(flag)?;
                this.write_scalar(result, dest)?;
            }
            "epoll_ctl" => {
                this.assert_target_os("illumos", "epoll_ctl");
                let [epfd, op, fd, event] =
                    this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;
                let result = this.epoll_ctl(epfd, op, fd, event)?;
                this.write_scalar(result, dest)?;
            }
            "epoll_wait" => {
                this.assert_target_os("illumos", "epoll_wait");
                let [epfd, events, maxevents, timeout] =
                    this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;
                this.epoll_wait(epfd, events, maxevents, timeout, dest)?;
            }
            "eventfd" => {
                this.assert_target_os("illumos", "eventfd");
                let [val, flag] = this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;
                let result = this.eventfd(val, flag)?;
                this.write_scalar(result, dest)?;
            }

            // Threading
            "pthread_setname_np" => {
                let [thread, name] =
                    this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;
                // THREAD_NAME_MAX allows a thread name of 31+1 length
                // https://github.com/illumos/illumos-gate/blob/7671517e13b8123748eda4ef1ee165c6d9dba7fe/usr/src/uts/common/sys/thread.h#L613
                let max_len = 32;
                // See https://illumos.org/man/3C/pthread_setname_np for the error codes.
                let res = match this.pthread_setname_np(
                    this.read_scalar(thread)?,
                    this.read_scalar(name)?,
                    max_len,
                    /* truncate */ false,
                )? {
                    ThreadNameResult::Ok => Scalar::from_u32(0),
                    ThreadNameResult::NameTooLong => this.eval_libc("ERANGE"),
                    ThreadNameResult::ThreadNotFound => this.eval_libc("ESRCH"),
                };
                this.write_scalar(res, dest)?;
            }
            "pthread_getname_np" => {
                let [thread, name, len] =
                    this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;
                // See https://illumos.org/man/3C/pthread_getname_np for the error codes.
                let res = match this.pthread_getname_np(
                    this.read_scalar(thread)?,
                    this.read_scalar(name)?,
                    this.read_scalar(len)?,
                    /* truncate */ false,
                )? {
                    ThreadNameResult::Ok => Scalar::from_u32(0),
                    ThreadNameResult::NameTooLong => this.eval_libc("ERANGE"),
                    ThreadNameResult::ThreadNotFound => this.eval_libc("ESRCH"),
                };
                this.write_scalar(res, dest)?;
            }

            // File related shims
            "stat" | "stat64" => {
                let [path, buf] = this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;
                let result = this.macos_fbsd_solarish_stat(path, buf)?;
                this.write_scalar(result, dest)?;
            }
            "lstat" | "lstat64" => {
                let [path, buf] = this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;
                let result = this.macos_fbsd_solarish_lstat(path, buf)?;
                this.write_scalar(result, dest)?;
            }
            "fstat" | "fstat64" => {
                let [fd, buf] = this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;
                let result = this.macos_fbsd_solarish_fstat(fd, buf)?;
                this.write_scalar(result, dest)?;
            }
            "readdir" => {
                let [dirp] = this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;
                let result = this.linux_solarish_readdir64("dirent", dirp)?;
                this.write_scalar(result, dest)?;
            }

            // Sockets and pipes
            "__xnet_socketpair" => {
                let [domain, type_, protocol, sv] =
                    this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;
                let result = this.socketpair(domain, type_, protocol, sv)?;
                this.write_scalar(result, dest)?;
            }

            // Miscellaneous
            "___errno" => {
                let [] = this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;
                let errno_place = this.last_error_place()?;
                this.write_scalar(errno_place.to_ref(this).to_scalar(), dest)?;
            }

            "stack_getbounds" => {
                let [stack] = this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;
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
                    this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;
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

                let cpus = this.deref_pointer_as(cpus, this.machine.layouts.u32)?;
                this.write_scalar(Scalar::from_u32(this.machine.num_cpus), &cpus)?;
                this.write_null(dest)?;
            }

            "__sysconf_xpg7" => {
                let [val] = this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;
                let result = this.sysconf(val)?;
                this.write_scalar(result, dest)?;
            }

            _ => return interp_ok(EmulateItemResult::NotSupported),
        }
        interp_ok(EmulateItemResult::NeedsReturn)
    }
}
