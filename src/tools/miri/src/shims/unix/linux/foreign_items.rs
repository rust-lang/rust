use rustc_span::Symbol;
use rustc_target::spec::abi::Abi;

use self::shims::unix::linux::epoll::EvalContextExt as _;
use self::shims::unix::linux::eventfd::EvalContextExt as _;
use self::shims::unix::linux::mem::EvalContextExt as _;
use self::shims::unix::linux::sync::futex;
use crate::helpers::check_min_arg_count;
use crate::machine::{SIGRTMAX, SIGRTMIN};
use crate::shims::unix::*;
use crate::*;

// The documentation of glibc complains that the kernel never exposes
// TASK_COMM_LEN through the headers, so it's assumed to always be 16 bytes
// long including a null terminator.
const TASK_COMM_LEN: usize = 16;

pub fn is_dyn_sym(name: &str) -> bool {
    matches!(name, "statx")
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

        // See `fn emulate_foreign_item_inner` in `shims/foreign_items.rs` for the general pattern.

        match link_name.as_str() {
            // File related shims
            "readdir64" => {
                let [dirp] = this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;
                let result = this.linux_readdir64(dirp)?;
                this.write_scalar(result, dest)?;
            }
            "sync_file_range" => {
                let [fd, offset, nbytes, flags] =
                    this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;
                let result = this.sync_file_range(fd, offset, nbytes, flags)?;
                this.write_scalar(result, dest)?;
            }
            "statx" => {
                let [dirfd, pathname, flags, mask, statxbuf] =
                    this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;
                let result = this.linux_statx(dirfd, pathname, flags, mask, statxbuf)?;
                this.write_scalar(result, dest)?;
            }

            // epoll, eventfd
            "epoll_create1" => {
                let [flag] = this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;
                let result = this.epoll_create1(flag)?;
                this.write_scalar(result, dest)?;
            }
            "epoll_ctl" => {
                let [epfd, op, fd, event] =
                    this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;
                let result = this.epoll_ctl(epfd, op, fd, event)?;
                this.write_scalar(result, dest)?;
            }
            "epoll_wait" => {
                let [epfd, events, maxevents, timeout] =
                    this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;
                this.epoll_wait(epfd, events, maxevents, timeout, dest)?;
            }
            "eventfd" => {
                let [val, flag] =
                    this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;
                let result = this.eventfd(val, flag)?;
                this.write_scalar(result, dest)?;
            }

            // Threading
            "pthread_setname_np" => {
                let [thread, name] =
                    this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;
                let res = this.pthread_setname_np(
                    this.read_scalar(thread)?,
                    this.read_scalar(name)?,
                    TASK_COMM_LEN,
                )?;
                let res = if res { Scalar::from_u32(0) } else { this.eval_libc("ERANGE") };
                this.write_scalar(res, dest)?;
            }
            "pthread_getname_np" => {
                let [thread, name, len] =
                    this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;
                // The function's behavior isn't portable between platforms.
                // In case of glibc, the length of the output buffer must
                // be not shorter than TASK_COMM_LEN.
                let len = this.read_scalar(len)?;
                let res = if len.to_target_usize(this)? >= TASK_COMM_LEN as u64
                    && this.pthread_getname_np(
                        this.read_scalar(thread)?,
                        this.read_scalar(name)?,
                        len,
                        /* truncate*/ false,
                    )? {
                    Scalar::from_u32(0)
                } else {
                    this.eval_libc("ERANGE")
                };
                this.write_scalar(res, dest)?;
            }
            "gettid" => {
                let [] = this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;
                let result = this.linux_gettid()?;
                this.write_scalar(result, dest)?;
            }

            // Dynamically invoked syscalls
            "syscall" => {
                // We do not use `check_shim` here because `syscall` is variadic. The argument
                // count is checked bellow.
                this.check_abi_and_shim_symbol_clash(abi, Abi::C { unwind: false }, link_name)?;
                // The syscall variadic function is legal to call with more arguments than needed,
                // extra arguments are simply ignored. The important check is that when we use an
                // argument, we have to also check all arguments *before* it to ensure that they
                // have the right type.

                let sys_getrandom = this.eval_libc("SYS_getrandom").to_target_usize(this)?;
                let sys_futex = this.eval_libc("SYS_futex").to_target_usize(this)?;
                let sys_eventfd2 = this.eval_libc("SYS_eventfd2").to_target_usize(this)?;

                let [op] = check_min_arg_count("syscall", args)?;
                match this.read_target_usize(op)? {
                    // `libc::syscall(NR_GETRANDOM, buf.as_mut_ptr(), buf.len(), GRND_NONBLOCK)`
                    // is called if a `HashMap` is created the regular way (e.g. HashMap<K, V>).
                    num if num == sys_getrandom => {
                        // Used by getrandom 0.1
                        // The first argument is the syscall id, so skip over it.
                        let [_, ptr, len, flags] =
                            check_min_arg_count("syscall(SYS_getrandom, ...)", args)?;

                        let ptr = this.read_pointer(ptr)?;
                        let len = this.read_target_usize(len)?;
                        // The only supported flags are GRND_RANDOM and GRND_NONBLOCK,
                        // neither of which have any effect on our current PRNG.
                        // See <https://github.com/rust-lang/rust/pull/79196> for a discussion of argument sizes.
                        let _flags = this.read_scalar(flags)?.to_i32()?;

                        this.gen_random(ptr, len)?;
                        this.write_scalar(Scalar::from_target_usize(len, this), dest)?;
                    }
                    // `futex` is used by some synchronization primitives.
                    num if num == sys_futex => {
                        futex(this, args, dest)?;
                    }
                    num if num == sys_eventfd2 => {
                        let [_, initval, flags] =
                            check_min_arg_count("syscall(SYS_evetfd2, ...)", args)?;

                        let result = this.eventfd(initval, flags)?;
                        this.write_int(result.to_i32()?, dest)?;
                    }
                    num => {
                        throw_unsup_format!("syscall: unsupported syscall number {num}");
                    }
                }
            }

            // Miscellaneous
            "mmap64" => {
                let [addr, length, prot, flags, fd, offset] =
                    this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;
                let offset = this.read_scalar(offset)?.to_i64()?;
                let ptr = this.mmap(addr, length, prot, flags, fd, offset.into())?;
                this.write_scalar(ptr, dest)?;
            }
            "mremap" => {
                let [old_address, old_size, new_size, flags] =
                    this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;
                let ptr = this.mremap(old_address, old_size, new_size, flags)?;
                this.write_scalar(ptr, dest)?;
            }
            "__errno_location" => {
                let [] = this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;
                let errno_place = this.last_error_place()?;
                this.write_scalar(errno_place.to_ref(this).to_scalar(), dest)?;
            }
            "__libc_current_sigrtmin" => {
                let [] = this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;

                this.write_int(SIGRTMIN, dest)?;
            }
            "__libc_current_sigrtmax" => {
                let [] = this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;

                this.write_int(SIGRTMAX, dest)?;
            }

            // Incomplete shims that we "stub out" just to get pre-main initialization code to work.
            // These shims are enabled only when the caller is in the standard library.
            "pthread_getattr_np" if this.frame_in_std() => {
                let [_thread, _attr] =
                    this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;
                this.write_null(dest)?;
            }

            _ => return interp_ok(EmulateItemResult::NotSupported),
        };

        interp_ok(EmulateItemResult::NeedsReturn)
    }
}
