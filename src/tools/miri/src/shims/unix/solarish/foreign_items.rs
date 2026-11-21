use rustc_middle::ty::Ty;
use rustc_span::Symbol;
use rustc_target::callconv::FnAbi;
use rustc_target::spec::Os;

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
        let ecx = self.eval_context_mut();
        match link_name.as_str() {
            // epoll, eventfd (NOT available on Solaris!)
            "epoll_create1" => {
                ecx.assert_target_os(Os::Illumos, "epoll_create1");
                let [flag] = &ecx
                    .check_shim_sig(shim_sig!(extern "C" fn(i32) -> i32), (link_name, abi, args))?;
                let result = ecx.epoll_create1(flag)?;
                ecx.write_scalar(result, dest)?;
            }
            "epoll_ctl" => {
                ecx.assert_target_os(Os::Illumos, "epoll_ctl");
                let [epfd, op, fd, event] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(i32, i32, i32, *libc::epoll_event) -> i32),
                    (link_name, abi, args),
                )?;
                let result = ecx.epoll_ctl(epfd, op, fd, event)?;
                ecx.write_scalar(result, dest)?;
            }
            "epoll_wait" => {
                ecx.assert_target_os(Os::Illumos, "epoll_wait");
                let [epfd, events, maxevents, timeout] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(i32, *libc::epoll_event, i32, i32) -> i32),
                    (link_name, abi, args),
                )?;
                ecx.epoll_wait(epfd, events, maxevents, timeout, dest)?;
            }
            "eventfd" => {
                ecx.assert_target_os(Os::Illumos, "eventfd");
                let [val, flag] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(u32, i32) -> i32),
                    (link_name, abi, args),
                )?;
                let result = ecx.eventfd(val, flag)?;
                ecx.write_scalar(result, dest)?;
            }

            // Threading
            "pthread_setname_np" => {
                let [thread, name] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(libc::pthread_t, *_) -> i32),
                    (link_name, abi, args),
                )?;
                // THREAD_NAME_MAX allows a thread name of 31+1 length
                // https://github.com/illumos/illumos-gate/blob/7671517e13b8123748eda4ef1ee165c6d9dba7fe/usr/src/uts/common/sys/thread.h#L613
                let max_len = 32;
                // See https://illumos.org/man/3C/pthread_setname_np for the error codes.
                let res = match ecx.pthread_setname_np(
                    ecx.read_scalar(thread)?,
                    ecx.read_scalar(name)?,
                    max_len,
                    /* truncate */ false,
                )? {
                    ThreadNameResult::Ok => Scalar::from_u32(0),
                    ThreadNameResult::NameTooLong => ecx.eval_libc("ERANGE"),
                };
                ecx.write_scalar(res, dest)?;
            }
            "pthread_getname_np" => {
                let [thread, name, len] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(libc::pthread_t, *_, usize) -> i32),
                    (link_name, abi, args),
                )?;
                // See https://illumos.org/man/3C/pthread_getname_np for the error codes.
                let res = match ecx.pthread_getname_np(
                    ecx.read_scalar(thread)?,
                    ecx.read_scalar(name)?,
                    ecx.read_scalar(len)?,
                    /* truncate */ false,
                )? {
                    ThreadNameResult::Ok => Scalar::from_u32(0),
                    ThreadNameResult::NameTooLong => ecx.eval_libc("ERANGE"),
                };
                ecx.write_scalar(res, dest)?;
            }

            // Sockets and pipes
            "__xnet_socketpair" => {
                let [domain, type_, protocol, sv] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(i32, i32, i32, *i32) -> i32),
                    (link_name, abi, args),
                )?;
                let result = ecx.socketpair(domain, type_, protocol, sv)?;
                ecx.write_scalar(result, dest)?;
            }

            // Network sockets
            "__xnet_socket" | "__xnet7_socket" => {
                let [domain, type_, protocol] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(i32, i32, i32) -> i32),
                    (link_name, abi, args),
                )?;
                let result = ecx.socket(domain, type_, protocol)?;
                ecx.write_scalar(result, dest)?;
            }
            "__xnet_bind" => {
                let [socket, address, address_len] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(i32, *libc::sockaddr, libc::socklen_t) -> i32),
                    (link_name, abi, args),
                )?;
                let result = ecx.bind(socket, address, address_len)?;
                ecx.write_scalar(result, dest)?;
            }
            "__xnet_connect" => {
                let [socket, address, address_len] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(i32, *libc::sockaddr, libc::socklen_t) -> i32),
                    (link_name, abi, args),
                )?;
                ecx.connect(socket, address, address_len, dest)?;
            }
            "__xnet_sendto" => {
                let [socket, buffer, length, flags, dest_addr, dest_addr_len] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(i32, *_, libc::size_t, i32, *libc::sockaddr, libc::socklen_t) -> libc::ssize_t),
                    (link_name, abi, args),
                )?;
                ecx.send(socket, buffer, length, flags, Some((dest_addr, dest_addr_len)), dest)?;
            }
            "__xnet_getaddrinfo" => {
                let [node, service, hints, res] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(*_, *_, *_, **_) -> i32),
                    (link_name, abi, args),
                )?;
                let result = ecx.getaddrinfo(node, service, hints, res)?;
                ecx.write_scalar(result, dest)?;
            }
            "__xnet_getsockopt" => {
                let [socket, level, option_name, option_value, option_len] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(i32, i32, i32, *_, *libc::socklen_t) -> i32),
                    (link_name, abi, args),
                )?;
                let result =
                    ecx.getsockopt(socket, level, option_name, option_value, option_len)?;
                ecx.write_scalar(result, dest)?;
            }

            // Miscellaneous
            "___errno" => {
                let [] =
                    &ecx.check_shim_sig(shim_sig!(extern "C" fn() -> *_), (link_name, abi, args))?;
                let errno_place = ecx.last_error_place()?;
                ecx.write_scalar(errno_place.to_ref(ecx).to_scalar(), dest)?;
            }

            "stack_getbounds" => {
                // FIXME: This does not have a direct test (#3179).
                let [stack] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(*libc::stack_t) -> i32),
                    (link_name, abi, args),
                )?;
                let stack = ecx.deref_pointer(stack)?;

                ecx.write_int_fields_named(
                    &[
                        ("ss_sp", ecx.machine.stack_addr.into()),
                        ("ss_size", ecx.machine.stack_size.into()),
                        // field set to 0 means not in an alternate signal stack
                        // https://docs.oracle.com/cd/E86824_01/html/E54766/stack-getbounds-3c.html
                        ("ss_flags", 0),
                    ],
                    &stack,
                )?;

                ecx.write_int(0, dest)?;
            }

            "pset_info" => {
                // FIXME: This does not have a direct test (#3179).
                let [pset, tpe, cpus, list] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(i32, *_, *u32, *_) -> i32),
                    (link_name, abi, args),
                )?;
                // We do not need to handle the current process cpu mask, available_parallelism
                // implementation pass null anyway. We only care for the number of
                // cpus.
                // https://docs.oracle.com/cd/E88353_01/html/E37841/pset-info-2.html

                let pset = ecx.read_scalar(pset)?.to_i32()?;
                let tpe = ecx.read_pointer(tpe)?;
                let list = ecx.read_pointer(list)?;

                let ps_myid = ecx.eval_libc_i32("PS_MYID");
                if ps_myid != pset {
                    throw_unsup_format!("pset_info is only supported with pset==PS_MYID");
                }

                if !ecx.ptr_is_null(tpe)? {
                    throw_unsup_format!("pset_info is only supported with type==NULL");
                }

                if !ecx.ptr_is_null(list)? {
                    throw_unsup_format!("pset_info is only supported with list==NULL");
                }

                let cpus = ecx.deref_pointer(cpus)?;
                ecx.write_scalar(Scalar::from_u32(ecx.machine.num_cpus), &cpus)?;
                ecx.write_int(0, dest)?;
            }

            "__sysconf_xpg7" => {
                let [val] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(i32) -> isize),
                    (link_name, abi, args),
                )?;
                let result = ecx.sysconf(val)?;
                ecx.write_scalar(result, dest)?;
            }

            _ => return interp_ok(EmulateItemResult::NotSupported),
        }
        interp_ok(EmulateItemResult::NeedsReturn)
    }
}
