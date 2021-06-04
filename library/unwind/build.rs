use std::env;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    let target = env::var("TARGET").expect("TARGET was not set");

    if cfg!(target_os = "linux") && cfg!(feature = "system-llvm-libunwind") {
        // linking for Linux is handled in lib.rs
        return;
    }

    if cfg!(feature = "llvm-libunwind")
        && ((target.contains("linux") && !target.contains("musl")) || target.contains("fuchsia"))
    {
        // Build the unwinding from libunwind C/C++ source code.
        llvm_libunwind::compile();
    } else if target.contains("x86_64-fortanix-unknown-sgx") {
        llvm_libunwind::compile();
    } else if target.contains("linux") {
        // linking for Linux is handled in lib.rs
        if target.contains("musl") {
            llvm_libunwind::compile();
        } else if target.contains("android") {
            let build = cc::Build::new();

            // Since ndk r23 beta 3 `libgcc` was replaced with `libunwind` thus
            // check if we have `libunwind` available and if so use it. Otherwise
            // fall back to `libgcc` to support older ndk versions.
            let has_unwind =
                build.is_flag_supported("-lunwind").expect("Unable to invoke compiler");

            if has_unwind {
                println!("cargo:rustc-link-lib=unwind");
            } else {
                println!("cargo:rustc-link-lib=gcc");
            }
        }
    } else if target.contains("freebsd") {
        println!("cargo:rustc-link-lib=gcc_s");
    } else if target.contains("netbsd") {
        println!("cargo:rustc-link-lib=gcc_s");
    } else if target.contains("openbsd") {
        if target.contains("sparc64") {
            println!("cargo:rustc-link-lib=gcc");
        } else {
            println!("cargo:rustc-link-lib=c++abi");
        }
    } else if target.contains("solaris") {
        println!("cargo:rustc-link-lib=gcc_s");
    } else if target.contains("illumos") {
        println!("cargo:rustc-link-lib=gcc_s");
    } else if target.contains("dragonfly") {
        println!("cargo:rustc-link-lib=gcc_pic");
    } else if target.contains("pc-windows-gnu") {
        // This is handled in the target spec with late_link_args_[static|dynamic]
    } else if target.contains("uwp-windows-gnu") {
        println!("cargo:rustc-link-lib=unwind");
    } else if target.contains("fuchsia") {
        println!("cargo:rustc-link-lib=unwind");
    } else if target.contains("haiku") {
        println!("cargo:rustc-link-lib=gcc_s");
    } else if target.contains("redox") {
        // redox is handled in lib.rs
    }
}

mod llvm_libunwind {
    use std::env;
    use std::path::Path;

    /// Compile the libunwind C/C++ source code.
    pub fn compile() {
        let target = env::var("TARGET").expect("TARGET was not set");
        let target_env = env::var("CARGO_CFG_TARGET_ENV").unwrap();
        let mut cc_cfg = cc::Build::new();
        let mut cpp_cfg = cc::Build::new();
        let root = Path::new("../../src/llvm-project/libunwind");

        cpp_cfg.cpp(true);
        cpp_cfg.cpp_set_stdlib(None);
        cpp_cfg.flag("-nostdinc++");
        cpp_cfg.flag("-fno-exceptions");
        cpp_cfg.flag("-fno-rtti");
        cpp_cfg.flag_if_supported("-fvisibility-global-new-delete-hidden");

        // Don't set this for clang
        // By default, Clang builds C code in GNU C17 mode.
        // By default, Clang builds C++ code according to the C++98 standard,
        // with many C++11 features accepted as extensions.
        if cpp_cfg.get_compiler().is_like_gnu() {
            cpp_cfg.flag("-std=c++11");
            cc_cfg.flag("-std=c99");
        }

        if target.contains("x86_64-fortanix-unknown-sgx") || target_env == "musl" {
            // use the same GCC C compiler command to compile C++ code so we do not need to setup the
            // C++ compiler env variables on the builders.
            // Don't set this for clang++, as clang++ is able to compile this without libc++.
            if cpp_cfg.get_compiler().is_like_gnu() {
                cpp_cfg.cpp(false);
            }
        }

        for cfg in [&mut cc_cfg, &mut cpp_cfg].iter_mut() {
            cfg.warnings(false);
            cfg.flag("-fstrict-aliasing");
            cfg.flag("-funwind-tables");
            cfg.flag("-fvisibility=hidden");
            cfg.define("_LIBUNWIND_DISABLE_VISIBILITY_ANNOTATIONS", None);
            cfg.include(root.join("include"));
            cfg.cargo_metadata(false);

            if target.contains("x86_64-fortanix-unknown-sgx") {
                cfg.static_flag(true);
                cfg.opt_level(3);
                cfg.flag("-fno-stack-protector");
                cfg.flag("-ffreestanding");
                cfg.flag("-fexceptions");

                // easiest way to undefine since no API available in cc::Build to undefine
                cfg.flag("-U_FORTIFY_SOURCE");
                cfg.define("_FORTIFY_SOURCE", "0");
                cfg.define("RUST_SGX", "1");
                cfg.define("__NO_STRING_INLINES", None);
                cfg.define("__NO_MATH_INLINES", None);
                cfg.define("_LIBUNWIND_IS_BAREMETAL", None);
                cfg.define("__LIBUNWIND_IS_NATIVE_ONLY", None);
                cfg.define("NDEBUG", None);
            }
        }

        let mut c_sources = vec![
            "Unwind-sjlj.c",
            "UnwindLevel1-gcc-ext.c",
            "UnwindLevel1.c",
            "UnwindRegistersRestore.S",
            "UnwindRegistersSave.S",
        ];

        let cpp_sources = vec!["Unwind-EHABI.cpp", "Unwind-seh.cpp", "libunwind.cpp"];
        let cpp_len = cpp_sources.len();

        if target.contains("x86_64-fortanix-unknown-sgx") {
            c_sources.push("UnwindRustSgx.c");
        }

        for src in c_sources {
            cc_cfg.file(root.join("src").join(src).canonicalize().unwrap());
        }

        for src in cpp_sources {
            cpp_cfg.file(root.join("src").join(src).canonicalize().unwrap());
        }

        let out_dir = env::var("OUT_DIR").unwrap();
        println!("cargo:rustc-link-search=native={}", &out_dir);

        cpp_cfg.compile("unwind-cpp");

        let mut count = 0;
        for entry in std::fs::read_dir(&out_dir).unwrap() {
            let obj = entry.unwrap().path().canonicalize().unwrap();
            if let Some(ext) = obj.extension() {
                if ext == "o" {
                    cc_cfg.object(&obj);
                    count += 1;
                }
            }
        }
        assert_eq!(cpp_len, count, "Can't get object files from {:?}", &out_dir);
        cc_cfg.compile("unwind");
    }
}
