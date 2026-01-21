use crate::{is_arm64ec, is_win7, is_windows, is_windows_msvc, target, uname};

fn get_windows_msvc_libs() -> Vec<&'static str> {
    let mut libs =
        vec!["ws2_32.lib", "userenv.lib", "bcrypt.lib", "ntdll.lib", "synchronization.lib"];
    if is_win7() {
        libs.push("advapi32.lib");
    }
    libs
}

/// `EXTRACFLAGS`
pub fn extra_c_flags() -> Vec<&'static str> {
    if is_windows() {
        if is_windows_msvc() {
            let mut args = get_windows_msvc_libs();
            if is_arm64ec() {
                args.push("/arm64EC");
            }
            args
        } else {
            vec!["-lws2_32", "-luserenv", "-lbcrypt", "-lntdll", "-lsynchronization"]
        }
    } else {
        // For cross-compilation targets, we need to check the target, not the host
        let target_triple = target();
        if target_triple.contains("hexagon") {
            // Hexagon targets need unwind support but don't have some Linux libraries
            vec!["-lunwind", "-lclang_rt.builtins-hexagon"]
        } else {
            // For host-based detection, fall back to uname() for non-cross compilation
            match uname() {
                n if n.contains("Darwin") => vec!["-lresolv"],
                n if n.contains("FreeBSD") => vec!["-lm", "-lpthread", "-lgcc_s"],
                n if n.contains("SunOS") => {
                    vec!["-lm", "-lpthread", "-lposix4", "-lsocket", "-lresolv"]
                }
                n if n.contains("OpenBSD") => vec!["-lm", "-lpthread", "-lc++abi"],
                _ => vec!["-lm", "-lrt", "-ldl", "-lpthread"],
            }
        }
    }
}

pub fn extra_linker_flags() -> Vec<&'static str> {
    if is_windows_msvc() {
        let mut args = get_windows_msvc_libs();
        if is_arm64ec() {
            args.push("/MACHINE:ARM64EC");
        }
        args
    } else {
        vec![]
    }
}

/// `EXTRACXXFLAGS`
pub fn extra_cxx_flags() -> Vec<&'static str> {
    if is_windows() {
        if is_windows_msvc() { vec![] } else { vec!["-lstdc++"] }
    } else {
        match &uname()[..] {
            "Darwin" => vec!["-lc++"],
            "FreeBSD" | "SunOS" | "OpenBSD" => vec![],
            _ => vec!["-lstdc++"],
        }
    }
}
