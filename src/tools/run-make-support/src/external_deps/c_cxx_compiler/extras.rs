use crate::{is_msvc, is_win7, is_windows, uname};

/// `EXTRACFLAGS`
pub fn extra_c_flags() -> Vec<&'static str> {
    if is_windows() {
        if is_msvc() {
            let mut libs =
                vec!["ws2_32.lib", "userenv.lib", "bcrypt.lib", "ntdll.lib", "synchronization.lib"];
            if is_win7() {
                libs.push("advapi32.lib");
            }
            libs
        } else {
            vec!["-lws2_32", "-luserenv", "-lbcrypt", "-lntdll", "-lsynchronization"]
        }
    } else {
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

/// `EXTRACXXFLAGS`
pub fn extra_cxx_flags() -> Vec<&'static str> {
    if is_windows() {
        if is_msvc() { vec![] } else { vec!["-lstdc++"] }
    } else {
        match &uname()[..] {
            "Darwin" => vec!["-lc++"],
            "FreeBSD" | "SunOS" | "OpenBSD" => vec![],
            _ => vec!["-lstdc++"],
        }
    }
}
