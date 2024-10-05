use rustc_span::Symbol;
use rustc_target::spec::abi::Abi;

use self::shims::unix::linux::epoll::EvalContextExt as _;
use self::shims::unix::linux::eventfd::EvalContextExt as _;
use self::shims::unix::linux::mem::EvalContextExt as _;
use self::shims::unix::linux::sync::futex;
use crate::machine::{SIGRTMAX, SIGRTMIN};
use crate::shims::unix::*;
use crate::*;

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
                let max_len = 16;
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

                if args.is_empty() {
                    throw_ub_format!(
                        "incorrect number of arguments for syscall: got 0, expected at least 1"
                    );
                }
                match this.read_target_usize(&args[0])? {
                    // `libc::syscall(NR_GETRANDOM, buf.as_mut_ptr(), buf.len(), GRND_NONBLOCK)`
                    // is called if a `HashMap` is created the regular way (e.g. HashMap<K, V>).
                    id if id == sys_getrandom => {
                        // Used by getrandom 0.1
                        // The first argument is the syscall id, so skip over it.
                        let [_, ptr, len, flags, ..] = args else {
                            throw_ub_format!(
                                "incorrect number of arguments for `getrandom` syscall: got {}, expected at least 4",
                                args.len()
                            );
                        };

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
                    id if id == sys_futex => {
                        futex(this, &args[1..], dest)?;
                    }
                    id => {
                        this.handle_unsupported_foreign_item(format!(
                            "can't execute syscall with ID {id}"
                        ))?;
                        return interp_ok(EmulateItemResult::AlreadyJumped);
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
