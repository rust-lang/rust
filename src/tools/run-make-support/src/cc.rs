use std::path::Path;

use crate::command::Command;
use crate::{bin_name, cygpath_windows, env_var, is_msvc, is_windows, uname};

/// Construct a new platform-specific C compiler invocation.
///
/// WARNING: This means that what flags are accepted by the underlying C compiler is
/// platform- AND compiler-specific. Consult the relevant docs for `gcc`, `clang` and `mvsc`.
#[track_caller]
pub fn cc() -> Cc {
    Cc::new()
}

/// A platform-specific C compiler invocation builder. The specific C compiler used is
/// passed down from compiletest.
#[derive(Debug)]
#[must_use]
pub struct Cc {
    cmd: Command,
}

crate::impl_common_helpers!(Cc);

impl Cc {
    /// Construct a new platform-specific C compiler invocation.
    ///
    /// WARNING: This means that what flags are accepted by the underlying C compile is
    /// platform- AND compiler-specific. Consult the relevant docs for `gcc`, `clang` and `mvsc`.
    #[track_caller]
    pub fn new() -> Self {
        let compiler = env_var("CC");

        let mut cmd = Command::new(compiler);

        let default_cflags = env_var("CC_DEFAULT_FLAGS");
        for flag in default_cflags.split(char::is_whitespace) {
            cmd.arg(flag);
        }

        Self { cmd }
    }

    /// Specify path of the input file.
    pub fn input<P: AsRef<Path>>(&mut self, path: P) -> &mut Self {
        self.cmd.arg(path.as_ref());
        self
    }

    /// Adds directories to the list that the linker searches for libraries.
    /// Equivalent to `-L`.
    pub fn library_search_path<P: AsRef<Path>>(&mut self, path: P) -> &mut Self {
        self.cmd.arg("-L");
        self.cmd.arg(path.as_ref());
        self
    }

    /// Specify `-o` or `-Fe`/`-Fo` depending on platform/compiler.
    pub fn out_exe(&mut self, name: &str) -> &mut Self {
        // Ref: tools.mk (irrelevant lines omitted):
        //
        // ```makefile
        // ifdef IS_MSVC
        //     OUT_EXE=-Fe:`cygpath -w $(TMPDIR)/$(call BIN,$(1))` \
        //         -Fo:`cygpath -w $(TMPDIR)/$(1).obj`
        // else
        //     OUT_EXE=-o $(TMPDIR)/$(1)
        // endif
        // ```

        if is_msvc() {
            let fe_path = cygpath_windows(bin_name(name));
            let fo_path = cygpath_windows(format!("{name}.obj"));
            self.cmd.arg(format!("-Fe:{fe_path}"));
            self.cmd.arg(format!("-Fo:{fo_path}"));
        } else {
            self.cmd.arg("-o");
            self.cmd.arg(name);
        }

        self
    }

    /// Specify path of the output binary.
    pub fn output<P: AsRef<Path>>(&mut self, path: P) -> &mut Self {
        self.cmd.arg("-o");
        self.cmd.arg(path.as_ref());
        self
    }
}

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
