use std::env;
use std::ffi::{OsStr, OsString};
use std::fmt::Display;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

const OPTIONAL_COMPONENTS: &[&str] = &[
    "x86",
    "arm",
    "aarch64",
    "amdgpu",
    "avr",
    "loongarch",
    "m68k",
    "csky",
    "mips",
    "powerpc",
    "systemz",
    "jsbackend",
    "webassembly",
    "msp430",
    "sparc",
    "nvptx",
    "hexagon",
    "riscv",
    "xtensa",
    "bpf",
];

const REQUIRED_COMPONENTS: &[&str] =
    &["ipo", "bitreader", "bitwriter", "linker", "asmparser", "lto", "coverage", "instrumentation"];

fn detect_llvm_link() -> (&'static str, &'static str) {
    // Force the link mode we want, preferring static by default, but
    // possibly overridden by `configure --enable-llvm-link-shared`.
    if tracked_env_var_os("LLVM_LINK_SHARED").is_some() {
        ("dylib", "--link-shared")
    } else {
        ("static", "--link-static")
    }
}

// Because Cargo adds the compiler's dylib path to our library search path, llvm-config may
// break: the dylib path for the compiler, as of this writing, contains a copy of the LLVM
// shared library, which means that when our freshly built llvm-config goes to load it's
// associated LLVM, it actually loads the compiler's LLVM. In particular when building the first
// compiler (i.e., in stage 0) that's a problem, as the compiler's LLVM is likely different from
// the one we want to use. As such, we restore the environment to what bootstrap saw. This isn't
// perfect -- we might actually want to see something from Cargo's added library paths -- but
// for now it works.
fn restore_library_path() {
    let key = tracked_env_var_os("REAL_LIBRARY_PATH_VAR").expect("REAL_LIBRARY_PATH_VAR");
    if let Some(env) = tracked_env_var_os("REAL_LIBRARY_PATH") {
        unsafe {
            env::set_var(&key, env);
        }
    } else {
        unsafe {
            env::remove_var(&key);
        }
    }
}

/// Reads an environment variable and adds it to dependencies.
/// Supposed to be used for all variables except those set for build scripts by cargo
/// <https://doc.rust-lang.org/cargo/reference/environment-variables.html#environment-variables-cargo-sets-for-build-scripts>
fn tracked_env_var_os<K: AsRef<OsStr> + Display>(key: K) -> Option<OsString> {
    println!("cargo:rerun-if-env-changed={key}");
    env::var_os(key)
}

fn rerun_if_changed_anything_in_dir(dir: &Path) {
    let mut stack = dir
        .read_dir()
        .unwrap()
        .map(|e| e.unwrap())
        .filter(|e| &*e.file_name() != ".git")
        .collect::<Vec<_>>();
    while let Some(entry) = stack.pop() {
        let path = entry.path();
        if entry.file_type().unwrap().is_dir() {
            stack.extend(path.read_dir().unwrap().map(|e| e.unwrap()));
        } else {
            println!("cargo:rerun-if-changed={}", path.display());
        }
    }
}

#[track_caller]
fn output(cmd: &mut Command) -> String {
    let output = match cmd.stderr(Stdio::inherit()).output() {
        Ok(status) => status,
        Err(e) => {
            println!("\n\nfailed to execute command: {cmd:?}\nerror: {e}\n\n");
            std::process::exit(1);
        }
    };
    if !output.status.success() {
        panic!(
            "command did not execute successfully: {:?}\n\
             expected success, got: {}",
            cmd, output.status
        );
    }
    String::from_utf8(output.stdout).unwrap()
}

fn main() {
    for component in REQUIRED_COMPONENTS.iter().chain(OPTIONAL_COMPONENTS.iter()) {
        println!("cargo:rustc-check-cfg=cfg(llvm_component,values(\"{component}\"))");
    }

    if tracked_env_var_os("RUST_CHECK").is_some() {
        // If we're just running `check`, there's no need for LLVM to be built.
        return;
    }

    restore_library_path();

    let llvm_config =
        PathBuf::from(tracked_env_var_os("LLVM_CONFIG").expect("LLVM_CONFIG was not set"));

    println!("cargo:rerun-if-changed={}", llvm_config.display());

    // Test whether we're cross-compiling LLVM. This is a pretty rare case
    // currently where we're producing an LLVM for a different platform than
    // what this build script is currently running on.
    //
    // In that case, there's no guarantee that we can actually run the target,
    // so the build system works around this by giving us the LLVM_CONFIG for
    // the host platform. This only really works if the host LLVM and target
    // LLVM are compiled the same way, but for us that's typically the case.
    //
    // We *want* detect this cross compiling situation by asking llvm-config
    // what its host-target is. If that's not the TARGET, then we're cross
    // compiling. Unfortunately `llvm-config` seems either be buggy, or we're
    // misconfiguring it, because the `i686-pc-windows-gnu` build of LLVM will
    // report itself with a `--host-target` of `x86_64-pc-windows-gnu`. This
    // tricks us into thinking we're doing a cross build when we aren't, so
    // havoc ensues.
    //
    // In any case, if we're cross compiling, this generally just means that we
    // can't trust all the output of llvm-config because it might be targeted
    // for the host rather than the target. As a result a bunch of blocks below
    // are gated on `if !is_crossed`
    let target = env::var("TARGET").expect("TARGET was not set");
    let host = env::var("HOST").expect("HOST was not set");
    let is_crossed = target != host;

    let components = output(Command::new(&llvm_config).arg("--components"));
    let mut components = components.split_whitespace().collect::<Vec<_>>();
    components.retain(|c| OPTIONAL_COMPONENTS.contains(c) || REQUIRED_COMPONENTS.contains(c));

    for component in REQUIRED_COMPONENTS {
        if !components.contains(component) {
            panic!("require llvm component {component} but wasn't found");
        }
    }

    for component in components.iter() {
        println!("cargo:rustc-cfg=llvm_component=\"{component}\"");
    }

    // Link in our own LLVM shims, compiled with the same flags as LLVM
    let mut cmd = Command::new(&llvm_config);
    cmd.arg("--cxxflags");
    let cxxflags = output(&mut cmd);
    let mut cfg = cc::Build::new();
    cfg.warnings(false);
    for flag in cxxflags.split_whitespace() {
        // Ignore flags like `-m64` when we're doing a cross build
        if is_crossed && flag.starts_with("-m") {
            continue;
        }

        if flag.starts_with("-flto") {
            continue;
        }

        // -Wdate-time is not supported by the netbsd cross compiler
        if is_crossed && target.contains("netbsd") && flag.contains("date-time") {
            continue;
        }

        // Include path contains host directory, replace it with target
        if is_crossed && flag.starts_with("-I") {
            cfg.flag(&flag.replace(&host, &target));
            continue;
        }

        cfg.flag(flag);
    }

    for component in &components {
        let mut flag = String::from("LLVM_COMPONENT_");
        flag.push_str(&component.to_uppercase());
        cfg.define(&flag, None);
    }

    if tracked_env_var_os("LLVM_ENZYME").is_some() {
        cfg.define("ENZYME", None);
    }

    if tracked_env_var_os("LLVM_RUSTLLVM").is_some() {
        cfg.define("LLVM_RUSTLLVM", None);
    }

    if tracked_env_var_os("LLVM_ASSERTIONS").is_none() {
        cfg.define("NDEBUG", None);
    }

    rerun_if_changed_anything_in_dir(Path::new("llvm-wrapper"));
    cfg.file("llvm-wrapper/PassWrapper.cpp")
        .file("llvm-wrapper/RustWrapper.cpp")
        .file("llvm-wrapper/ArchiveWrapper.cpp")
        .file("llvm-wrapper/CoverageMappingWrapper.cpp")
        .file("llvm-wrapper/SymbolWrapper.cpp")
        .file("llvm-wrapper/Linker.cpp")
        .cpp(true)
        .cpp_link_stdlib(None) // we handle this below
        .compile("llvm-wrapper");

    let (llvm_kind, llvm_link_arg) = detect_llvm_link();

    // Link in all LLVM libraries, if we're using the "wrong" llvm-config then
    // we don't pick up system libs because unfortunately they're for the host
    // of llvm-config, not the target that we're attempting to link.
    let mut cmd = Command::new(&llvm_config);
    cmd.arg(llvm_link_arg).arg("--libs");

    // Don't link system libs if cross-compiling unless targeting Windows.
    // On Windows system DLLs aren't linked directly, instead import libraries are used.
    // These import libraries are independent of the host.
    if !is_crossed || target.contains("windows") {
        cmd.arg("--system-libs");
    }

    // We need libkstat for getHostCPUName on SPARC builds.
    // See also: https://github.com/llvm/llvm-project/issues/64186
    if target.starts_with("sparcv9") && target.contains("solaris") {
        println!("cargo:rustc-link-lib=kstat");
    }

    if (target.starts_with("arm") && !target.contains("freebsd")) && !target.contains("ohos")
        || target.starts_with("mips-")
        || target.starts_with("mipsel-")
        || target.starts_with("powerpc-")
        || target.starts_with("sparc-")
    {
        // 32-bit targets need to link libatomic.
        println!("cargo:rustc-link-lib=atomic");
    } else if target.contains("windows-gnu") {
        println!("cargo:rustc-link-lib=shell32");
        println!("cargo:rustc-link-lib=uuid");
    } else if target.contains("haiku")
        || target.contains("darwin")
        || (is_crossed && (target.contains("dragonfly") || target.contains("solaris")))
        || target.contains("cygwin")
    {
        println!("cargo:rustc-link-lib=z");
    } else if target.contains("netbsd") {
        // On NetBSD/i386, gcc and g++ is built for i486 (to maximize backward compat)
        // However, LLVM insists on using 64-bit atomics.
        // This gives rise to a need to link rust itself with -latomic for these targets
        if target.starts_with("i586") || target.starts_with("i686") {
            println!("cargo:rustc-link-lib=atomic");
        }
        println!("cargo:rustc-link-lib=z");
        println!("cargo:rustc-link-lib=execinfo");
    }
    cmd.args(&components);

    for lib in output(&mut cmd).split_whitespace() {
        let mut is_static = false;
        let name = if let Some(stripped) = lib.strip_prefix("-l") {
            stripped
        } else if let Some(stripped) = lib.strip_prefix('-') {
            stripped
        } else if Path::new(lib).exists() {
            // On MSVC llvm-config will print the full name to libraries, but
            // we're only interested in the name part
            // On Unix when we get a static library llvm-config will print the
            // full name and we *are* interested in the path, but we need to
            // handle it separately. For example, when statically linking to
            // libzstd llvm-config will output something like
            //   -lrt -ldl -lm -lz /usr/local/lib/libzstd.a -lxml2
            // and we transform the zstd part into
            //   cargo:rustc-link-search-native=/usr/local/lib
            //   cargo:rustc-link-lib=static=zstd
            let path = Path::new(lib);
            if lib.ends_with(".a") {
                is_static = true;
                println!("cargo:rustc-link-search=native={}", path.parent().unwrap().display());
                let name = path.file_stem().unwrap().to_str().unwrap();
                name.trim_start_matches("lib")
            } else {
                let name = path.file_name().unwrap().to_str().unwrap();
                name.trim_end_matches(".lib")
            }
        } else if lib.ends_with(".lib") {
            // Some MSVC libraries just come up with `.lib` tacked on, so chop
            // that off
            lib.trim_end_matches(".lib")
        } else {
            continue;
        };

        // Don't need or want this library, but LLVM's CMake build system
        // doesn't provide a way to disable it, so filter it here even though we
        // may or may not have built it. We don't reference anything from this
        // library and it otherwise may just pull in extra dependencies on
        // libedit which we don't want
        if name == "LLVMLineEditor" {
            continue;
        }

        let kind = if name.starts_with("LLVM") {
            llvm_kind
        } else if is_static {
            "static"
        } else {
            "dylib"
        };
        println!("cargo:rustc-link-lib={kind}={name}");
    }

    // LLVM ldflags
    //
    // If we're a cross-compile of LLVM then unfortunately we can't trust these
    // ldflags (largely where all the LLVM libs are located). Currently just
    // hack around this by replacing the host triple with the target and pray
    // that those -L directories are the same!
    let mut cmd = Command::new(&llvm_config);
    cmd.arg(llvm_link_arg).arg("--ldflags");
    for lib in output(&mut cmd).split_whitespace() {
        if is_crossed {
            if let Some(stripped) = lib.strip_prefix("-LIBPATH:") {
                println!("cargo:rustc-link-search=native={}", stripped.replace(&host, &target));
            } else if let Some(stripped) = lib.strip_prefix("-L") {
                println!("cargo:rustc-link-search=native={}", stripped.replace(&host, &target));
            }
        } else if let Some(stripped) = lib.strip_prefix("-LIBPATH:") {
            println!("cargo:rustc-link-search=native={stripped}");
        } else if let Some(stripped) = lib.strip_prefix("-l") {
            println!("cargo:rustc-link-lib={stripped}");
        } else if let Some(stripped) = lib.strip_prefix("-L") {
            println!("cargo:rustc-link-search=native={stripped}");
        }
    }

    // Some LLVM linker flags (-L and -l) may be needed even when linking
    // rustc_llvm, for example when using static libc++, we may need to
    // manually specify the library search path and -ldl -lpthread as link
    // dependencies.
    let llvm_linker_flags = tracked_env_var_os("LLVM_LINKER_FLAGS");
    if let Some(s) = llvm_linker_flags {
        for lib in s.into_string().unwrap().split_whitespace() {
            if let Some(stripped) = lib.strip_prefix("-l") {
                println!("cargo:rustc-link-lib={stripped}");
            } else if let Some(stripped) = lib.strip_prefix("-L") {
                println!("cargo:rustc-link-search=native={stripped}");
            }
        }
    }

    let llvm_static_stdcpp = tracked_env_var_os("LLVM_STATIC_STDCPP");
    let llvm_use_libcxx = tracked_env_var_os("LLVM_USE_LIBCXX");

    let stdcppname = if target.contains("openbsd") {
        if target.contains("sparc64") { "estdc++" } else { "c++" }
    } else if target.contains("darwin")
        || target.contains("freebsd")
        || target.contains("windows-gnullvm")
        || target.contains("aix")
        || target.contains("ohos")
    {
        "c++"
    } else if target.contains("netbsd") && llvm_static_stdcpp.is_some() {
        // NetBSD uses a separate library when relocation is required
        "stdc++_p"
    } else if llvm_use_libcxx.is_some() {
        "c++"
    } else {
        "stdc++"
    };

    // RISC-V GCC erroneously requires libatomic for sub-word
    // atomic operations. Some BSD uses Clang as its system
    // compiler and provides no libatomic in its base system so
    // does not want this.
    if target.starts_with("riscv") && !target.contains("freebsd") && !target.contains("openbsd") {
        println!("cargo:rustc-link-lib=atomic");
    }

    // C++ runtime library
    if !target.contains("msvc") {
        if let Some(s) = llvm_static_stdcpp {
            assert!(!cxxflags.contains("stdlib=libc++"));
            let path = PathBuf::from(s);
            println!("cargo:rustc-link-search=native={}", path.parent().unwrap().display());
            if target.contains("windows") {
                println!("cargo:rustc-link-lib=static:-bundle={stdcppname}");
            } else {
                println!("cargo:rustc-link-lib=static={stdcppname}");
            }
        } else if cxxflags.contains("stdlib=libc++") {
            println!("cargo:rustc-link-lib=c++");
        } else {
            println!("cargo:rustc-link-lib={stdcppname}");
        }
    }

    // libc++abi and libunwind have to be specified explicitly on AIX.
    if target.contains("aix") {
        println!("cargo:rustc-link-lib=c++abi");
        println!("cargo:rustc-link-lib=unwind");
    }

    // Libstdc++ depends on pthread which Rust doesn't link on MinGW
    // since nothing else requires it.
    if target.ends_with("windows-gnu") {
        println!("cargo:rustc-link-lib=static:-bundle=pthread");
    }
}
