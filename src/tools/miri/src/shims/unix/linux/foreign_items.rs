use rustc_span::Symbol;
use rustc_target::spec::abi::Abi;

use crate::machine::SIGRTMAX;
use crate::machine::SIGRTMIN;
use crate::*;
use shims::foreign_items::EmulateByNameResult;
use shims::unix::fs::EvalContextExt as _;
use shims::unix::linux::fd::EvalContextExt as _;
use shims::unix::linux::mem::EvalContextExt as _;
use shims::unix::linux::sync::futex;
use shims::unix::sync::EvalContextExt as _;
use shims::unix::thread::EvalContextExt as _;

impl<'mir, 'tcx: 'mir> EvalContextExt<'mir, 'tcx> for crate::MiriInterpCx<'mir, 'tcx> {}
pub trait EvalContextExt<'mir, 'tcx: 'mir>: crate::MiriInterpCxExt<'mir, 'tcx> {
    fn emulate_foreign_item_by_name(
        &mut self,
        link_name: Symbol,
        abi: Abi,
        args: &[OpTy<'tcx, Provenance>],
        dest: &PlaceTy<'tcx, Provenance>,
    ) -> InterpResult<'tcx, EmulateByNameResult<'mir, 'tcx>> {
        let this = self.eval_context_mut();

        // See `fn emulate_foreign_item_by_name` in `shims/foreign_items.rs` for the general pattern.

        match link_name.as_str() {
            // errno
            "__errno_location" => {
                let [] = this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;
                let errno_place = this.last_error_place()?;
                this.write_scalar(errno_place.to_ref(this).to_scalar(), dest)?;
            }

            // File related shims (but also see "syscall" below for statx)
            "readdir64" => {
                let [dirp] = this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;
                let result = this.linux_readdir64(dirp)?;
                this.write_scalar(result, dest)?;
            }
            // Linux-only
            "sync_file_range" => {
                let [fd, offset, nbytes, flags] =
                    this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;
                let result = this.sync_file_range(fd, offset, nbytes, flags)?;
                this.write_scalar(result, dest)?;
            }
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
                let result = this.epoll_wait(epfd, events, maxevents, timeout)?;
                this.write_scalar(result, dest)?;
            }
            "eventfd" => {
                let [val, flag] =
                    this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;
                let result = this.eventfd(val, flag)?;
                this.write_scalar(result, dest)?;
            }
            "mremap" => {
                let [old_address, old_size, new_size, flags] =
                    this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;
                let ptr = this.mremap(old_address, old_size, new_size, flags)?;
                this.write_scalar(ptr, dest)?;
            }
            "socketpair" => {
                let [domain, type_, protocol, sv] =
                    this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;

                let result = this.socketpair(domain, type_, protocol, sv)?;
                this.write_scalar(result, dest)?;
            }
            "__libc_current_sigrtmin" => {
                let [] = this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;

                this.write_scalar(Scalar::from_i32(SIGRTMIN), dest)?;
            }
            "__libc_current_sigrtmax" => {
                let [] = this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;

                this.write_scalar(Scalar::from_i32(SIGRTMAX), dest)?;
            }

            // Threading
            "pthread_condattr_setclock" => {
                let [attr, clock_id] =
                    this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;
                let result = this.pthread_condattr_setclock(attr, clock_id)?;
                this.write_scalar(result, dest)?;
            }
            "pthread_condattr_getclock" => {
                let [attr, clock_id] =
                    this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;
                let result = this.pthread_condattr_getclock(attr, clock_id)?;
                this.write_scalar(result, dest)?;
            }
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

                let sys_statx = this.eval_libc("SYS_statx").to_target_usize(this)?;

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
                        // The first argument is the syscall id, so skip over it.
                        if args.len() < 4 {
                            throw_ub_format!(
                                "incorrect number of arguments for `getrandom` syscall: got {}, expected at least 4",
                                args.len()
                            );
                        }
                        getrandom(this, &args[1], &args[2], &args[3], dest)?;
                    }
                    // `statx` is used by `libstd` to retrieve metadata information on `linux`
                    // instead of using `stat`,`lstat` or `fstat` as on `macos`.
                    id if id == sys_statx => {
                        // The first argument is the syscall id, so skip over it.
                        if args.len() < 6 {
                            throw_ub_format!(
                                "incorrect number of arguments for `statx` syscall: got {}, expected at least 6",
                                args.len()
                            );
                        }
                        let result =
                            this.linux_statx(&args[1], &args[2], &args[3], &args[4], &args[5])?;
                        this.write_scalar(Scalar::from_target_isize(result.into(), this), dest)?;
                    }
                    // `futex` is used by some synchronization primitives.
                    id if id == sys_futex => {
                        futex(this, &args[1..], dest)?;
                    }
                    id => {
                        this.handle_unsupported(format!("can't execute syscall with ID {id}"))?;
                        return Ok(EmulateByNameResult::AlreadyJumped);
                    }
                }
            }

            // Miscellaneous
            "getrandom" => {
                let [ptr, len, flags] =
                    this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;
                getrandom(this, ptr, len, flags, dest)?;
            }
            "sched_getaffinity" => {
                let [pid, cpusetsize, mask] =
                    this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;
                this.read_scalar(pid)?.to_i32()?;
                this.read_target_usize(cpusetsize)?;
                this.deref_operand_as(mask, this.libc_ty_layout("cpu_set_t"))?;
                // FIXME: we just return an error; `num_cpus` then falls back to `sysconf`.
                let einval = this.eval_libc("EINVAL");
                this.set_last_error(einval)?;
                this.write_scalar(Scalar::from_i32(-1), dest)?;
            }

            // Incomplete shims that we "stub out" just to get pre-main initialization code to work.
            // These shims are enabled only when the caller is in the standard library.
            "pthread_getattr_np" if this.frame_in_std() => {
                let [_thread, _attr] =
                    this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;
                this.write_null(dest)?;
            }

            _ => return Ok(EmulateByNameResult::NotSupported),
        };

        Ok(EmulateByNameResult::NeedsJumping)
    }
}

// Shims the linux `getrandom` syscall.
fn getrandom<'tcx>(
    this: &mut MiriInterpCx<'_, 'tcx>,
    ptr: &OpTy<'tcx, Provenance>,
    len: &OpTy<'tcx, Provenance>,
    flags: &OpTy<'tcx, Provenance>,
    dest: &PlaceTy<'tcx, Provenance>,
) -> InterpResult<'tcx> {
    let ptr = this.read_pointer(ptr)?;
    let len = this.read_target_usize(len)?;

    // The only supported flags are GRND_RANDOM and GRND_NONBLOCK,
    // neither of which have any effect on our current PRNG.
    // See <https://github.com/rust-lang/rust/pull/79196> for a discussion of argument sizes.
    let _flags = this.read_scalar(flags)?.to_i32();

    this.gen_random(ptr, len)?;
    this.write_scalar(Scalar::from_target_usize(len, this), dest)?;
    Ok(())
}
