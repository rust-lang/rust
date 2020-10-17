use std::env;

use crate::spec::{LinkArgs, TargetOptions};

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
    // Here we detect what version is being requested, defaulting to 10.7. ELF
    // TLS is flagged as enabled if it looks to be supported.
    let version = macos_deployment_target();

    TargetOptions {
        // macOS has -dead_strip, which doesn't rely on function_sections
        function_sections: false,
        dynamic_linking: true,
        executables: true,
        target_family: Some("unix".to_string()),
        is_like_osx: true,
        dwarf_version: Some(2),
        has_rpath: true,
        dll_prefix: "lib".to_string(),
        dll_suffix: ".dylib".to_string(),
        archive_format: "darwin".to_string(),
        pre_link_args: LinkArgs::new(),
        has_elf_tls: version >= (10, 7),
        abi_return_struct_as_int: true,
        emit_debug_gdb_scripts: false,
        eh_frame_header: false,

        // This environment variable is pretty magical but is intended for
        // producing deterministic builds. This was first discovered to be used
        // by the `ar` tool as a way to control whether or not mtime entries in
        // the archive headers were set to zero or not. It appears that
        // eventually the linker got updated to do the same thing and now reads
        // this environment variable too in recent versions.
        //
        // For some more info see the commentary on #47086
        link_env: vec![("ZERO_AR_DATE".to_string(), "1".to_string())],

        ..Default::default()
    }
}

fn macos_deployment_target() -> (u32, u32) {
    let deployment_target = env::var("MACOSX_DEPLOYMENT_TARGET").ok();
    let version = deployment_target
        .as_ref()
        .and_then(|s| {
            let mut i = s.splitn(2, '.');
            i.next().and_then(|a| i.next().map(|b| (a, b)))
        })
        .and_then(|(a, b)| a.parse::<u32>().and_then(|a| b.parse::<u32>().map(|b| (a, b))).ok());

    version.unwrap_or((10, 7))
}

pub fn macos_llvm_target(arch: &str) -> String {
    let (major, minor) = macos_deployment_target();
    format!("{}-apple-macosx{}.{}.0", arch, major, minor)
}

pub fn macos_link_env_remove() -> Vec<String> {
    let mut env_remove = Vec::with_capacity(2);
    // Remove the `SDKROOT` environment variable if it's clearly set for the wrong platform, which
    // may occur when we're linking a custom build script while targeting iOS for example.
    if let Ok(sdkroot) = env::var("SDKROOT") {
        if sdkroot.contains("iPhoneOS.platform") || sdkroot.contains("iPhoneSimulator.platform") {
            env_remove.push("SDKROOT".to_string())
        }
    }
    // Additionally, `IPHONEOS_DEPLOYMENT_TARGET` must not be set when using the Xcode linker at
    // "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/ld",
    // although this is apparently ignored when using the linker at "/usr/bin/ld".
    env_remove.push("IPHONEOS_DEPLOYMENT_TARGET".to_string());
    env_remove
}
