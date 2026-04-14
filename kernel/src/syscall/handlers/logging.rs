//! Logging and debug output syscalls

use super::copyin;
use crate::syscall::validate::validate_user_range;
use abi::errors::{Errno, SysResult};

pub fn sys_log_write(ptr: usize, len: usize, level_arg: usize) -> SysResult<usize> {
    let _ = validate_user_range(ptr, len, false)?;
    if len > 2048 {
        return Err(Errno::EINVAL);
    }

    let level = match level_arg {
        1 => crate::logging::LogLevel::Error,
        2 => crate::logging::LogLevel::Warn,
        3 => crate::logging::LogLevel::Info,
        4 => crate::logging::LogLevel::Debug,
        5 => crate::logging::LogLevel::Trace,
        _ => crate::logging::LogLevel::Info,
    };

    let mut buf = alloc::vec![0u8; len];
    unsafe {
        copyin(&mut buf[..len], ptr)?;
    }

    match core::str::from_utf8(&buf) {
        Ok(s) => {
            let s_trimmed = s.trim_end();

            let (provenance, msg_body) = if let Some(idx) = s_trimmed.find(": ") {
                let (prefix, rest) = s_trimmed.split_at(idx);
                if prefix.len() < 32 && !prefix.contains(char::is_whitespace) {
                    (prefix, &rest[2..])
                } else {
                    ("user.print", s_trimmed)
                }
            } else {
                ("user.print", s_trimmed)
            };

            crate::logging::_log_event(
                crate::logging::LogMetadata {
                    level,
                    file: "userspace",
                    line: 0,
                    module: "user",
                },
                provenance,
                format_args!("{}", msg_body),
                &[],
                &[],
            );
        }
        Err(_) => return Err(Errno::EINVAL),
    }

    Ok(len)
}

pub fn sys_debug_write(ptr: usize, len: usize) -> SysResult<usize> {
    sys_log_write(ptr, len, 4)
}
