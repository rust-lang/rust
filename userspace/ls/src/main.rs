#![no_std]
#![no_main]
extern crate alloc;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::prelude::v1::*;
use stem::syscall::{argv_get, exit, vfs_close, vfs_open, vfs_readdir, vfs_stat, vfs_write};

#[derive(Debug, Default)]
struct Flags {
    long: bool,
    recursive: bool,
    all: bool,
}

fn get_args() -> (Vec<String>, Flags) {
    let mut len = 0;
    if let Ok(l) = argv_get(&mut []) {
        len = l;
    }
    if len == 0 {
        return (Vec::new(), Flags::default());
    }
    let mut buf = alloc::vec![0u8; len];
    if argv_get(&mut buf).is_err() {
        return (Vec::new(), Flags::default());
    }

    let raw_args = stem::utils::parse_argv(&buf);
    let mut paths = Vec::new();
    let mut flags = Flags::default();

    // Skip arg[0] (command name)
    for arg_bytes in raw_args.into_iter().skip(1) {
        if let Ok(arg) = core::str::from_utf8(arg_bytes) {
            if arg.starts_with('-') && arg.len() > 1 {
                for c in arg.chars().skip(1) {
                    match c {
                        'l' => flags.long = true,
                        'R' => flags.recursive = true,
                        'a' => flags.all = true,
                        _ => {}
                    }
                }
            } else {
                paths.push(String::from(arg));
            }
        }
    }

    if paths.is_empty() {
        paths.push(String::from("."));
    }

    (paths, flags)
}

fn print(msg: &str) {
    let _ = vfs_write(1, msg.as_bytes());
}

fn format_mode(mode: u32) -> String {
    let mut s = String::with_capacity(10);
    let kind = mode & 0o170000;
    if kind == 0o040000 {
        s.push('d');
    } else if kind == 0o020000 {
        s.push('c');
    } else if kind == 0o010000 {
        s.push('p');
    } else {
        s.push('-');
    }

    let perms = mode & 0o777;
    for i in (0..3).rev() {
        let p = (perms >> (i * 3)) & 0o7;
        s.push(if p & 0o4 != 0 { 'r' } else { '-' });
        s.push(if p & 0o2 != 0 { 'w' } else { '-' });
        s.push(if p & 0o1 != 0 { 'x' } else { '-' });
    }
    s
}

const COLOR_DIR: &str = "\x1B[34m";
const COLOR_EXE: &str = "\x1B[32m";
const COLOR_DEV: &str = "\x1B[33m";
const COLOR_RESET: &str = "\x1B[0m";

fn list_path(path: &str, flags: &Flags, is_nested: bool) {
    stem::debug!("ls: listing path '{}'", path);
    if flags.recursive || is_nested {
        print(&alloc::format!("{}:\n", path));
    }

    let fd = match vfs_open(path, 0) {
        // O_RDONLY = 0
        Ok(fd) => fd,
        Err(e) => {
            stem::error!("ls: failed to open '{}': {:?}", path, e);
            print(&alloc::format!(
                "ls: cannot access '{}': No such file or directory\n",
                path
            ));
            return;
        }
    };

    let stat = match vfs_stat(fd) {
        Ok(s) => s,
        Err(_) => {
            let _ = vfs_close(fd);
            return;
        }
    };

    let _name_at_path = path.split('/').last().unwrap_or(path);
    let is_dir = (stat.mode & 0o170000) == 0o040000;
    let is_exe = (stat.mode & 0o111) != 0;
    let is_dev = (stat.mode & 0o020000) != 0 || (stat.mode & 0o060000) != 0;

    let color = if is_dir {
        COLOR_DIR
    } else if is_dev {
        COLOR_DEV
    } else if is_exe {
        COLOR_EXE
    } else {
        ""
    };

    if (stat.mode & 0o170000) != 0o040000 {
        // Not a directory, just print the file itself
        if flags.long {
            print(&alloc::format!(
                "{} {:8} {}{}{}\n",
                format_mode(stat.mode),
                stat.size,
                color,
                path,
                if color.is_empty() { "" } else { COLOR_RESET }
            ));
        } else {
            print(&alloc::format!("{}{}{}\n", color, path, if color.is_empty() { "" } else { COLOR_RESET }));
        }
        let _ = vfs_close(fd);
        return;
    }

    let mut entries = Vec::new();
    let mut buf = [0u8; 4096];
    loop {
        match vfs_readdir(fd, &mut buf) {
            Ok(0) => break,
            Ok(n) => {
                let mut offset = 0;
                while offset < n {
                    let mut end = offset;
                    while end < n && buf[end] != 0 {
                        end += 1;
                    }
                    if let Ok(name) = core::str::from_utf8(&buf[offset..end]) {
                        if !name.is_empty() {
                            if flags.all || !name.starts_with('.') {
                                entries.push(String::from(name));
                            }
                        }
                    }
                    offset = end + 1;
                }
            }
            Err(_) => break,
        }
    }
    let _ = vfs_close(fd);

    // Sort entries for consistency
    entries.sort();

    let mut subdirs = Vec::new();

    for name in entries {
        let mut full_path = String::from(path);
        if !full_path.ends_with('/') {
            full_path.push('/');
        }
        full_path.push_str(&name);

        match vfs_open(&full_path, 0) {
            Ok(child_fd) => {
                if let Ok(child_stat) = vfs_stat(child_fd) {
                    let c_is_dir = (child_stat.mode & 0o170000) == 0o040000;
                    let c_is_exe = (child_stat.mode & 0o111) != 0 && !c_is_dir;
                    let c_is_dev = (child_stat.mode & 0o020000) != 0 || (child_stat.mode & 0o060000) != 0;

                    let c_color = if c_is_dir {
                        COLOR_DIR
                    } else if c_is_dev {
                        COLOR_DEV
                    } else if c_is_exe {
                        COLOR_EXE
                    } else {
                        ""
                    };

                    if flags.long {
                        print(&alloc::format!(
                            "{} {:8} {}{}{}\n",
                            format_mode(child_stat.mode),
                            child_stat.size,
                            c_color,
                            name,
                            if c_color.is_empty() { "" } else { COLOR_RESET }
                        ));
                    } else {
                        print(&alloc::format!("{}{}{}  ", c_color, name, if c_color.is_empty() { "" } else { COLOR_RESET }));
                    }

                    if flags.recursive
                        && c_is_dir
                        && name != "."
                        && name != ".."
                    {
                        subdirs.push(full_path);
                    }
                }
                let _ = vfs_close(child_fd);
            }
            Err(_) => {
                if flags.long {
                    print(&alloc::format!("?--------- ?        {}\n", name));
                } else {
                    print(&alloc::format!("{}  ", name));
                }
            }
        }
    }

    if !flags.long {
        print("\n");
    }

    if flags.recursive && !subdirs.is_empty() {
        print("\n");
        for subdir in subdirs {
            list_path(&subdir, flags, true);
        }
    }
}

#[stem::main]
fn main(_arg: usize) -> ! {
    let (paths, flags) = get_args();

    for (i, path) in paths.iter().enumerate() {
        if i > 0 {
            print("\n");
        }
        list_path(path, &flags, paths.len() > 1);
    }

    exit(0)
}
