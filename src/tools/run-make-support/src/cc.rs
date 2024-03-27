use std::env;
use std::path::Path;
use std::process::{Command, Output};

use crate::{bin_name, cygpath_windows, handle_failed_output, is_msvc, is_windows, tmp_dir, uname};

/// Construct a new platform-specific C/C++ compiler invocation.
///
/// WARNING: This means that what flags are accepted by the underlying C/C++ compile is
/// platform- AND compiler-specific. Consult the relevant docs for `gcc`, `clang` and `mvsc`.
pub fn cc() -> Cc {
    Cc::new()
}

/// A platform-specific C/C++ compiler invocation builder. The specific C/C++ compiler used is
/// passed down from compiletest.
#[derive(Debug)]
pub struct Cc {
    cmd: Command,
}

impl Cc {
    /// Construct a new platform-specific C/C++ compiler invocation.
    ///
    /// WARNING: This means that what flags are accepted by the underlying C/C++ compile is
    /// platform- AND compiler-specific. Consult the relevant docs for `gcc`, `clang` and `mvsc`.
    pub fn new() -> Self {
        let var = env::var("CC").unwrap();
        // FIXME(jieyouxu): wouldn't this be less hacky if we passed the compiler and preset
        // flags separately?
        let (compiler, flags) = var
            .split_once(" ")
            .expect("expected CC to be a compiler followed by flags like `cc -funsafe-math`");
        let mut cmd = Command::new(compiler);
        for flag in flags.split(char::is_whitespace) {
            cmd.arg(flag);
        }
        Self { cmd }
    }

    /// Specify path of the input file.
    pub fn input<P: AsRef<Path>>(&mut self, path: P) -> &mut Self {
        self.cmd.arg(path.as_ref());
        self
    }

    /// Add a *platform-and-compiler-specific* argument. Please consult the docs for the various
    /// possible C/C++ compilers on the various platforms to check which arguments are legal for
    /// which compiler.
    pub fn arg(&mut self, flag: &str) -> &mut Self {
        self.cmd.arg(flag);
        self
    }

    /// Specify `-o` or `-Fe`/`-Fo` depending on platform/compiler. This assumes that the executable
    /// is under `$TMPDIR`.
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
            let fe_path = cygpath_windows(tmp_dir().join(bin_name(name)));
            let fo_path = cygpath_windows(tmp_dir().join(format!("{name}.obj")));
            self.cmd.arg(format!("-Fe:{fe_path}"));
            self.cmd.arg(format!("-Fo:{fo_path}"));
        } else {
            self.cmd.arg("-o");
            self.cmd.arg(tmp_dir().join(name));
        }

        self
    }

    /// Run the constructed C/C++ invocation command and assert that it is successfully run.
    #[track_caller]
    pub fn run(&mut self) -> Output {
        let caller_location = std::panic::Location::caller();
        let caller_line_number = caller_location.line();

        let output = self.cmd.output().unwrap();
        if !output.status.success() {
            handle_failed_output(&format!("{:#?}", self.cmd), output, caller_line_number);
        }
        output
    }

    /// Inspect what the underlying [`Command`] is up to the current construction.
    pub fn inspect(&mut self, f: impl FnOnce(&Command)) -> &mut Self {
        f(&self.cmd);
        self
    }
}

/// `EXTRACFLAGS`
pub fn extra_c_flags() -> String {
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
            "ws2_32.lib userenv.lib advapi32.lib bcrypt.lib ntdll.lib synchronization.lib"
                .to_string()
        } else {
            "-lws2_32 -luserenv -lbcrypt -lntdll -lsynchronization".to_string()
        }
    } else {
        match uname() {
            n if n.contains("Darwin") => "-lresolv".to_string(),
            n if n.contains("FreeBSD") => "-lm -lpthread -lgcc_s".to_string(),
            n if n.contains("SunOS") => "-lm -lpthread -lposix4 -lsocket -lresolv".to_string(),
            n if n.contains("OpenBSD") => "-lm -lpthread -lc++abi".to_string(),
            _ => "-lm -lrt -ldl -lpthread".to_string(),
        }
    }
}

/// `EXTRACXXFLAGS`
pub fn extra_cxx_flags() -> String {
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
        if is_msvc() { String::new() } else { "-lstdc++".to_string() }
    } else {
        match uname() {
            n if n.contains("Darwin") => "-lc++".to_string(),
            _ => "-lstdc++".to_string(),
        }
    }
}
