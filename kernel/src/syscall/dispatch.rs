use crate::syscall::handlers;

use abi::syscall::*;

pub fn dispatch(n: usize, args: [usize; 6]) -> isize {
    let syscall_id = n as u32;

    let result = match syscall_id {
        SYS_EXIT => handlers::sys_exit(args[0] as i32),
        SYS_REBOOT => handlers::sys_reboot(args[0]),
        SYS_READ => handlers::sys_read(args[0], args[1], args[2]),
        SYS_WRITE => handlers::sys_write(args[0], args[1], args[2]),
        SYS_DEBUG_WRITE => handlers::sys_debug_write(args[0], args[1]),
        SYS_LOG_WRITE => handlers::sys_log_write(args[0], args[1], args[2]),
        SYS_YIELD => handlers::sys_yield(),
        SYS_SLEEP_MS => handlers::sys_sleep_ms(args[0] as u64),
        SYS_SLEEP => handlers::sys_sleep_ns(args[0] as u64),
        SYS_TIME_MONOTONIC => handlers::sys_time_monotonic_ns(),
        SYS_TIME_NOW => handlers::sys_time_now(args[0] as u32, args[1]),
        SYS_TIME_ANCHOR => handlers::sys_time_anchor(args[0] as u64),
        SYS_DEVICE_CALL => handlers::sys_device_call(args[0]),
        SYS_SPAWN_THREAD => handlers::sys_spawn_thread(args[0], args[1]),
        SYS_SPAWN_PROCESS => handlers::sys_spawn_process(args[0], args[1], args[2]),
        SYS_GET_TID => handlers::sys_get_tid(),
        SYS_TASK_POLL => handlers::sys_task_poll(args[0]),
        SYS_TASK_KILL => handlers::sys_task_kill(args[0]),
        SYS_SET_PRIORITY => handlers::sys_set_priority(args[0], args[1]),
        SYS_TASK_DUMP => handlers::sys_task_dump(),
        SYS_GETPID => handlers::sys_getpid(),
        SYS_GETPPID => handlers::sys_getppid(),
        SYS_ARGV_GET => handlers::sys_argv_get(args[0], args[1]),
        SYS_ENV_GET => handlers::sys_env_get(args[0], args[1], args[2], args[3]),
        SYS_ENV_SET => handlers::sys_env_set(args[0], args[1], args[2], args[3]),
        SYS_ENV_UNSET => handlers::sys_env_unset(args[0], args[1]),
        SYS_ENV_LIST => handlers::sys_env_list(args[0], args[1]),
        SYS_AUXV_GET => handlers::sys_auxv_get(args[0], args[1]),
        SYS_SPAWN_PROCESS_EX => handlers::sys_spawn_process_ex(args[0], args[1]),
        SYS_ALLOC_STACK => handlers::sys_alloc_stack(args[0]),
        SYS_FUTEX_WAIT => handlers::sys_futex_wait(args[0], args[1] as u32, args[2] as u64),
        SYS_FUTEX_WAKE => handlers::sys_futex_wake(args[0], args[1] as u32),
        SYS_WAIT_MANY => {
            handlers::sys_wait_many(args[0], args[1], args[2], args[3], args[4] as u64)
        }
        SYS_VM_MAP => handlers::sys_vm_map(args[0], args[1]),
        SYS_VM_UNMAP => handlers::sys_vm_unmap(args[0], args[1]),
        SYS_VM_PROTECT => handlers::sys_vm_protect(args[0]),
        SYS_VM_ADVISE => handlers::sys_vm_advise(args[0]),
        SYS_VM_QUERY => handlers::sys_vm_query(args[0], args[1]),
        SYS_TASK_WAIT => handlers::sys_task_wait(args[0]),
        SYS_WAITPID => handlers::sys_waitpid(args[0], args[1], args[2]),
        SYS_TASK_EXEC => {
            handlers::sys_task_exec(args[0] as u32, args[1], args[2], args[3], args[4])
        }
        SYS_AVAILABLE_PARALLELISM => handlers::sys_available_parallelism(),
        SYS_TASK_SET_TLS_BASE => handlers::sys_task_set_tls_base(args[0]),
        SYS_TASK_GET_TLS_BASE => handlers::sys_task_get_tls_base(),
        SYS_TASK_INTERRUPT => handlers::sys_task_interrupt(args[0]),
        SYS_TASK_SET_NAME => handlers::sys_task_set_name(args[0], args[1]),

        // ── Signal management ─────────────────────────────────────────────
        SYS_KILL => handlers::sys_kill(args[0], args[1]),
        SYS_RAISE => handlers::sys_raise(args[0]),
        SYS_SIGACTION => handlers::sys_sigaction(args[0], args[1], args[2]),
        SYS_SIGPROCMASK => handlers::sys_sigprocmask(args[0], args[1], args[2]),
        SYS_SIGPENDING => handlers::sys_sigpending(args[0]),
        SYS_SIGSUSPEND => handlers::sys_sigsuspend(args[0]),
        // SYS_SIGRETURN is handled in flat.rs (needs raw frame pointer).
        SYS_SIGRETURN => Ok(0), // handled by kernel_dispatch_flat
        SYS_ALARM => handlers::sys_alarm(args[0]),
        SYS_PAUSE => handlers::sys_pause(),
        SYS_SETPGID => handlers::sys_setpgid(args[0], args[1]),
        SYS_GETPGRP => handlers::sys_getpgrp(),
        SYS_SETSID => handlers::sys_setsid(),

        SYS_CHANNEL_CREATE => handlers::sys_channel_create(args[0]),
        SYS_CHANNEL_SEND => handlers::sys_channel_send(args[0], args[1], args[2]),
        SYS_CHANNEL_SEND_ALL => handlers::sys_channel_send_all(args[0], args[1], args[2]),
        SYS_CHANNEL_RECV => handlers::sys_channel_recv(args[0], args[1], args[2]),
        SYS_CHANNEL_CLOSE => handlers::sys_channel_close(args[0]),
        // Deprecated: use SYS_FS_POLL after SYS_FD_FROM_HANDLE instead.
        SYS_CHANNEL_WAIT => handlers::sys_channel_wait(args[0], args[1], args[2]),
        SYS_CHANNEL_INFO => handlers::sys_channel_info(args[0]),
        SYS_CHANNEL_TRY_RECV => handlers::sys_channel_try_recv(args[0], args[1], args[2]),
        // Deprecated: use SYS_CHANNEL_SEND_MSG instead.
        SYS_CHANNEL_SEND_HANDLE => handlers::sys_channel_send_handle(args[0], args[1]),
        // Deprecated: use SYS_CHANNEL_RECV_MSG instead.
        SYS_CHANNEL_RECV_HANDLE => handlers::sys_channel_recv_handle(args[0], args[1]),
        SYS_CHANNEL_SEND_MSG => {
            handlers::sys_channel_send_msg(args[0], args[1], args[2], args[3], args[4])
        }
        SYS_CHANNEL_RECV_MSG => {
            handlers::sys_channel_recv_msg(args[0], args[1], args[2], args[3], args[4], args[5])
        }

        SYS_TRACE_READ => handlers::sys_trace_read(args[0], args[1]),
        SYS_CONSOLE_DISABLE => handlers::sys_console_disable(),

        SYS_DEVICE_CLAIM => handlers::sys_device_claim(args[0], args[1]),
        SYS_DEVICE_MAP_MMIO => handlers::sys_device_map_mmio(args[0], args[1]),
        SYS_DEVICE_IRQ_SUBSCRIBE => handlers::sys_device_irq_subscribe(args[0], args[1], args[2]),
        SYS_DEVICE_IOPORT_READ => handlers::sys_device_ioport(args[0], 0, false, args[1]),
        SYS_DEVICE_IOPORT_WRITE => handlers::sys_device_ioport(args[0], args[1], true, args[2]),
        SYS_DEVICE_ALLOC_DMA => handlers::sys_device_alloc_dma(args[0], args[1]),
        SYS_DEVICE_DMA_PHYS => handlers::sys_device_dma_phys(args[0]),
        SYS_DEVICE_IRQ_WAIT => handlers::sys_device_irq_wait(args[0], args[1], args[2]),

        SYS_MEMFD_CREATE => handlers::sys_memfd_create(args[0], args[1], args[2]),
        SYS_MEMFD_PHYS => handlers::sys_memfd_phys(args[0]),

        SYS_GETRANDOM => handlers::sys_getrandom(args[0], args[1]),
        SYS_ENTROPY_SEED => handlers::sys_entropy_seed(args[0], args[1]),
        SYS_LOG_SET_LEVEL => {
            crate::logging::set_log_level(args[0] as u8);
            Ok(0)
        }

        // ── VFS (janix) ───────────────────────────────────────────────────
        SYS_FS_OPEN => handlers::vfs::sys_fs_open(args[0], args[1], args[2]),
        SYS_FS_CLOSE => handlers::vfs::sys_fs_close(args[0]),
        SYS_FS_READ => handlers::vfs::sys_fs_read(args[0], args[1], args[2]),
        SYS_FS_WRITE => handlers::vfs::sys_fs_write(args[0], args[1], args[2]),
        SYS_FS_DUP => handlers::vfs::SYS_FS_DUP(args[0]),
        SYS_FS_DUP2 => handlers::vfs::SYS_FS_DUP2(args[0], args[1]),
        SYS_FS_RENAME => {
            handlers::vfs::sys_fs_rename(args[0], args[1], args[2], args[3], args[4], args[5])
        }
        SYS_PIPE => handlers::vfs::sys_pipe(args[0]),
        SYS_FS_UNLINK => handlers::vfs::sys_fs_unlink(args[0], args[1]),
        SYS_FS_MKDIR => handlers::vfs::sys_fs_mkdir(args[0], args[1]),
        SYS_FS_CHDIR => handlers::vfs::sys_fs_chdir(args[0], args[1]),
        SYS_FS_GETCWD => handlers::vfs::sys_fs_getcwd(args[0], args[1]),
        SYS_FS_MOUNT => handlers::vfs::sys_fs_mount(args[0], args[1], args[2]),
        SYS_FS_UMOUNT => handlers::vfs::sys_fs_umount(args[0], args[1]),
        SYS_FS_STAT => handlers::vfs::sys_fs_stat(args[0], args[1], args[2], args[3]),
        SYS_FS_READDIR => handlers::vfs::sys_fs_readdir(args[0], args[1], args[2]),
        SYS_FS_POLL => handlers::vfs::sys_fs_poll(args[0], args[1], args[2]),
        SYS_FS_SEEK => handlers::vfs::sys_fs_seek(args[0], args[1], args[2]),
        SYS_FS_WATCH_FD => handlers::vfs::sys_watch_fd(args[0], args[1], args[2]),
        SYS_FS_DEVICE_CALL => handlers::vfs::sys_fs_device_call(args[0], args[1]),
        SYS_FS_WATCH_PATH => handlers::vfs::sys_watch_path(args[0], args[1], args[2], args[3]),
        SYS_FD_FROM_HANDLE => handlers::vfs::sys_fd_from_handle(args[0]),
        SYS_FS_NOTIFY => handlers::vfs::sys_fs_notify(args[0], args[1], args[2]),
        SYS_FS_ISATTY => handlers::vfs::sys_fs_isatty(args[0]),
        SYS_FS_REALPATH => handlers::vfs::sys_fs_realpath(args[0], args[1], args[2], args[3]),
        SYS_FS_SYNC => handlers::vfs::sys_fs_sync(args[0]),
        SYS_FS_FTRUNCATE => handlers::vfs::sys_fs_ftruncate(args[0], args[1]),
        SYS_FS_FCNTL => handlers::vfs::sys_fs_fcntl(args[0], args[1], args[2]),
        SYS_FS_SYMLINK => handlers::vfs::sys_fs_symlink(args[0], args[1], args[2], args[3]),
        SYS_FS_READLINK => handlers::vfs::sys_fs_readlink(args[0], args[1], args[2], args[3]),
        SYS_FS_LINK => handlers::vfs::sys_fs_link(args[0], args[1], args[2], args[3]),
        SYS_FS_CHMOD => handlers::vfs::sys_fs_chmod(args[0], args[1], args[2]),
        SYS_FS_FCHMOD => handlers::vfs::sys_fs_fchmod(args[0], args[1]),
        SYS_FS_UTIMES => handlers::vfs::sys_fs_utimes(args[0], args[1], args[2], args[3]),
        SYS_FS_FUTIMES => handlers::vfs::sys_fs_futimes(args[0], args[1]),
        SYS_FS_LSTAT => handlers::vfs::sys_fs_lstat(args[0], args[1], args[2]),
        SYS_FS_READV => handlers::vfs::sys_fs_readv(args[0], args[1], args[2]),
        SYS_FS_WRITEV => handlers::vfs::sys_fs_writev(args[0], args[1], args[2]),
        SYS_FS_LUTIMES => handlers::vfs::sys_fs_lutimes(args[0], args[1], args[2]),
        SYS_FS_FLOCK => handlers::vfs::sys_fs_flock(args[0], args[1]),

        // ── Unix domain sockets ───────────────────────────────────────────
        SYS_SOCKET => handlers::sys_socket(args[0], args[1], args[2]),
        SYS_BIND => handlers::sys_bind(args[0], args[1], args[2]),
        SYS_LISTEN => handlers::sys_listen(args[0], args[1]),
        SYS_ACCEPT => handlers::sys_accept(args[0]),
        SYS_CONNECT => handlers::sys_connect(args[0], args[1], args[2]),
        SYS_SHUTDOWN => handlers::sys_shutdown(args[0], args[1]),
        SYS_SOCKETPAIR => handlers::sys_socketpair(args[0], args[1], args[2], args[3]),

        _ => Err(abi::errors::Errno::ENOSYS),
    };

    match result {
        Ok(val) => val as isize,
        Err(e) => -(e as isize),
    }
}
