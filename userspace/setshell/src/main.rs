#![no_std]
#![no_main]
extern crate alloc;

use alloc::string::{String, ToString};
use alloc::vec::Vec;
use stem::abi::syscall::vfs_flags;
use stem::syscall::{argv_get, exit, vfs_close, vfs_mkdir, vfs_open, vfs_stat, vfs_write};

const RUNTIME_SHELL_PATH: &str = "/run/sprout/shell";

fn get_args() -> Vec<String> {
    let mut len = 0;
    if let Ok(l) = argv_get(&mut []) {
        len = l;
    }
    if len == 0 {
        return Vec::new();
    }

    let mut buf = alloc::vec![0u8; len];
    if argv_get(&mut buf).is_err() {
        return Vec::new();
    }

    stem::utils::parse_argv(&buf)
        .into_iter()
        .skip(1)
        .filter_map(|b| core::str::from_utf8(b).ok().map(String::from))
        .collect()
}

fn print(msg: &str) {
    let _ = vfs_write(1, msg.as_bytes());
}

fn print_err(msg: &str) {
    let _ = vfs_write(2, msg.as_bytes());
}

fn ensure_parent_dirs() {
    let _ = vfs_mkdir("/run");
    let _ = vfs_mkdir("/run/sprout");
}

fn validate_shell_path(path: &str) -> Result<(), &'static str> {
    if path.is_empty() {
        return Err("empty path");
    }
    if !path.starts_with('/') {
        return Err("path must be absolute");
    }

    let fd = vfs_open(path, vfs_flags::O_RDONLY).map_err(|_| "shell binary not found")?;
    let stat = vfs_stat(fd).map_err(|_| "failed to stat shell binary")?;
    let _ = vfs_close(fd);

    let kind = stat.mode & 0o170000;
    if kind != 0o100000 {
        return Err("path is not a regular file");
    }

    Ok(())
}

fn write_runtime_shell(path: &str) -> Result<(), &'static str> {
    ensure_parent_dirs();

    let fd = vfs_open(
        RUNTIME_SHELL_PATH,
        vfs_flags::O_WRONLY | vfs_flags::O_CREAT | vfs_flags::O_TRUNC,
    )
    .map_err(|_| "failed to open /run/sprout/shell")?;

    let mut line = path.to_string();
    line.push('\n');
    let write_ok = vfs_write(fd, line.as_bytes()).is_ok();
    let _ = vfs_close(fd);

    if !write_ok {
        return Err("failed to write shell override");
    }

    Ok(())
}

#[stem::main]
fn main(_arg: usize) -> ! {
    let args = get_args();
    if args.len() != 1 {
        print("usage: setshell </bin/program>\n");
        exit(2)
    }

    let candidate = args[0].trim();
    if let Err(e) = validate_shell_path(candidate) {
        print_err(&alloc::format!("setshell: {}: {}\n", candidate, e));
        exit(1)
    }

    if let Err(e) = write_runtime_shell(candidate) {
        print_err(&alloc::format!("setshell: {}\n", e));
        exit(1)
    }

    print(&alloc::format!(
        "sprout shell set to '{}' (applies on next shell restart)\n",
        candidate
    ));
    exit(0)
}
