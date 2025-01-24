use crate::{is_msvc, is_windows, uname};

/// `EXTRACFLAGS`
pub fn extra_c_flags() -> Vec<&'static str> {
    // Adapted from tools.mk (trimmed):
    //
    // ```makefile
    // ifdef IS_WINDOWS
    //     ifdef IS_MSVC
    //         EXTRACFLAGS := ws2_32.lib userenv.lib advapi32.lib bcrypt.lib ntdll.lib synchronization.lib
    //     else
    //         EXTRACFLAGS := -lws2_32 -luserenv -lbcrypt -lntdll -lsynchronization
    //     endif
    // else
    //     ifeq ($(UNAME),Darwin)
    //         EXTRACFLAGS := -lresolv
    //     else
    //         ifeq ($(UNAME),FreeBSD)
    //             EXTRACFLAGS := -lm -lpthread -lgcc_s
    //         else
    //             ifeq ($(UNAME),SunOS)
    //                 EXTRACFLAGS := -lm -lpthread -lposix4 -lsocket -lresolv
    //             else
    //                 ifeq ($(UNAME),OpenBSD)
    //                     EXTRACFLAGS := -lm -lpthread -lc++abi
    //                 else
    //                     EXTRACFLAGS := -lm -lrt -ldl -lpthread
    //                 endif
    //             endif
    //         endif
    //     endif
    // endif
    // ```

    if is_windows() {
        if is_msvc() {
            vec![
                "ws2_32.lib",
                "userenv.lib",
                "advapi32.lib",
                "bcrypt.lib",
                "ntdll.lib",
                "synchronization.lib",
            ]
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
    // Adapted from tools.mk (trimmed):
    //
    // ```makefile
    // ifdef IS_WINDOWS
    //     ifdef IS_MSVC
    //     else
    //         EXTRACXXFLAGS := -lstdc++
    //     endif
    // else
    //     ifeq ($(UNAME),Darwin)
    //         EXTRACXXFLAGS := -lc++
    //     else
    //         ifeq ($(UNAME),FreeBSD)
    //         else
    //             ifeq ($(UNAME),SunOS)
    //             else
    //                 ifeq ($(UNAME),OpenBSD)
    //                 else
    //                     EXTRACXXFLAGS := -lstdc++
    //                 endif
    //             endif
    //         endif
    //     endif
    // endif
    // ```
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
