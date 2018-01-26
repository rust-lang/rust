// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::env;
use std::io;
use std::process::Command;
use std::str;

use target::TargetOptions;

pub fn xcrun(print_arg: &str, sdk_name: &str) -> io::Result<String> {
    Command::new("xcrun").arg(print_arg).arg("--sdk").arg(sdk_name).output().and_then(|output| {
        if output.status.success() {
            Ok(str::from_utf8(&output.stdout[..]).unwrap().trim().to_string())
        } else {
            let error = format!(
                "process exit with error: {}",
                str::from_utf8(&output.stderr[..]).unwrap(),
            );
            Err(io::Error::new(io::ErrorKind::Other, &error[..]))
        }
    })
}

pub fn get_sdk_root(sdk_name: &str) -> Result<String, String> {
    xcrun("--show-sdk-path", sdk_name).map_err(|e| {
        format!("failed to get {} SDK path: {}", sdk_name, e)
    })
}

pub fn get_sdk_version(sdk_name: &str) -> Result<String, String> {
    xcrun("--show-sdk-version", sdk_name).map_err(|e| {
        format!("failed to get {} SDK version: {}", sdk_name, e)
    })
}

pub fn get_deployment_target() -> String {
    env::var("MACOSX_DEPLOYMENT_TARGET").or_else(|_e| {
        get_sdk_version("macosx")
    }).unwrap_or("10.7".to_string())
}

pub fn opts() -> TargetOptions {
    // ELF TLS is only available in macOS 10.7+. If you try to compile for 10.6
    // either the linker will complain if it is used or the binary will end up
    // segfaulting at runtime when run on 10.6. Rust by default supports macOS
    // 10.7+, but there is a standard environment variable,
    // MACOSX_DEPLOYMENT_TARGET, which is used to signal targeting older
    // versions of macOS. For example compiling on 10.10 with
    // MACOSX_DEPLOYMENT_TARGET set to 10.6 will cause the linker to generate
    // warnings about the usage of ELF TLS.
    //
    // Here we detect what version is being requested. ELF TLS is flagged as
    // enabled if it looks to be supported.
    let deployment_target = get_deployment_target();
    let mut i = deployment_target.splitn(2, '.').map(|s| s.parse::<u32>().unwrap());
    let version = (i.next().unwrap(), i.next().unwrap());

    TargetOptions {
        // macOS has -dead_strip, which doesn't rely on function_sections
        function_sections: false,
        dynamic_linking: true,
        executables: true,
        target_family: Some("unix".to_string()),
        is_like_osx: true,
        has_rpath: true,
        dll_prefix: "lib".to_string(),
        dll_suffix: ".dylib".to_string(),
        archive_format: "bsd".to_string(),
        exe_allocation_crate: super::maybe_jemalloc(),
        has_elf_tls: version >= (10, 7),
        linker: "ld".to_string(),
        .. Default::default()
    }
}
