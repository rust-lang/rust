// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use PanicStrategy;
use LinkerFlavor;
use target::{LinkArgs, TargetOptions};
use std::default::Default;
use std::env;
use std::process::Command;

// Use GCC to locate code for crt* libraries from the host, not from L4Re. Note
// that a few files also come from L4Re, for these, the function shouldn't be
// used. This uses GCC for the location of the file, but GCC is required for L4Re anyway.
fn get_path_or(filename: &str) -> String {
    let child = Command::new("gcc")
        .arg(format!("-print-file-name={}", filename)).output()
        .expect("Failed to execute GCC");
    String::from_utf8(child.stdout)
        .expect("Couldn't read path from GCC").trim().into()
}

pub fn opts() -> TargetOptions {
    let l4re_lib_path = env::var_os("L4RE_LIBDIR").expect("Unable to find L4Re \
        library directory: L4RE_LIBDIR not set.").into_string().unwrap();
    let mut pre_link_args = LinkArgs::new();
    pre_link_args.insert(LinkerFlavor::Ld, vec![
        format!("-T{}/main_stat.ld", l4re_lib_path),
        "--defsym=__executable_start=0x01000000".to_string(),
        "--defsym=__L4_KIP_ADDR__=0x6ffff000".to_string(),
        format!("{}/crt1.o", l4re_lib_path),
        format!("{}/crti.o", l4re_lib_path),
        get_path_or("crtbeginT.o"),
    ]);
    let mut post_link_args = LinkArgs::new();
    post_link_args.insert(LinkerFlavor::Ld, vec![
        format!("{}/l4f/libpthread.a", l4re_lib_path),
        format!("{}/l4f/libc_be_sig.a", l4re_lib_path),
        format!("{}/l4f/libc_be_sig_noop.a", l4re_lib_path),
        format!("{}/l4f/libc_be_socket_noop.a", l4re_lib_path),
        format!("{}/l4f/libc_be_fs_noop.a", l4re_lib_path),
        format!("{}/l4f/libc_be_sem_noop.a", l4re_lib_path),
        format!("{}/l4f/libl4re-vfs.o.a", l4re_lib_path),
        format!("{}/l4f/lib4re.a", l4re_lib_path),
        format!("{}/l4f/lib4re-util.a", l4re_lib_path),
        format!("{}/l4f/libc_support_misc.a", l4re_lib_path),
        format!("{}/l4f/libsupc++.a", l4re_lib_path),
        format!("{}/l4f/lib4shmc.a", l4re_lib_path),
        format!("{}/l4f/lib4re-c.a", l4re_lib_path),
        format!("{}/l4f/lib4re-c-util.a", l4re_lib_path),
        get_path_or("libgcc_eh.a"),
        format!("{}/l4f/libdl.a", l4re_lib_path),
        "--start-group".to_string(),
        format!("{}/l4f/libl4util.a", l4re_lib_path),
        format!("{}/l4f/libc_be_l4re.a", l4re_lib_path),
        format!("{}/l4f/libuc_c.a", l4re_lib_path),
        format!("{}/l4f/libc_be_l4refile.a", l4re_lib_path),
        "--end-group".to_string(),
        format!("{}/l4f/libl4sys.a", l4re_lib_path),
        "-gc-sections".to_string(),
        get_path_or("crtend.o"),
        format!("{}/crtn.o", l4re_lib_path),
    ]);

    TargetOptions {
        executables: true,
        has_elf_tls: false,
        exe_allocation_crate: None,
        panic_strategy: PanicStrategy::Abort,
        linker: "ld".to_string(),
        pre_link_args,
        post_link_args,
        target_family: Some("unix".to_string()),
        .. Default::default()
    }
}
