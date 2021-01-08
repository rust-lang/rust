use std::env;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    let target = env::var("TARGET").expect("TARGET was not set");

    if cfg!(feature = "system-llvm-libunwind") {
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
        }
    } else if target.contains("freebsd") {
        println!("cargo:rustc-link-lib=gcc_s");
    } else if target.contains("rumprun") {
        println!("cargo:rustc-link-lib=unwind");
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
        let target_vendor = env::var("CARGO_CFG_TARGET_VENDOR").unwrap();
        let target_endian_little = env::var("CARGO_CFG_TARGET_ENDIAN").unwrap() != "big";
        let cfg = &mut cc::Build::new();

        cfg.cpp(true);
        cfg.cpp_set_stdlib(None);
        cfg.warnings(false);

        // libunwind expects a __LITTLE_ENDIAN__ macro to be set for LE archs, cf. #65765
        if target_endian_little {
            cfg.define("__LITTLE_ENDIAN__", Some("1"));
        }

        if target_env == "msvc" {
            // Don't pull in extra libraries on MSVC
            cfg.flag("/Zl");
            cfg.flag("/EHsc");
            cfg.define("_CRT_SECURE_NO_WARNINGS", None);
            cfg.define("_LIBUNWIND_DISABLE_VISIBILITY_ANNOTATIONS", None);
        } else if target.contains("x86_64-fortanix-unknown-sgx") {
            cfg.cpp(false);

            cfg.static_flag(true);
            cfg.opt_level(3);

            cfg.flag("-nostdinc++");
            cfg.flag("-fno-exceptions");
            cfg.flag("-fno-rtti");
            cfg.flag("-fstrict-aliasing");
            cfg.flag("-funwind-tables");
            cfg.flag("-fvisibility=hidden");
            cfg.flag("-fno-stack-protector");
            cfg.flag("-ffreestanding");
            cfg.flag("-fexceptions");

            // easiest way to undefine since no API available in cc::Build to undefine
            cfg.flag("-U_FORTIFY_SOURCE");
            cfg.define("_FORTIFY_SOURCE", "0");

            cfg.flag_if_supported("-fvisibility-global-new-delete-hidden");

            cfg.define("_LIBUNWIND_DISABLE_VISIBILITY_ANNOTATIONS", None);
            cfg.define("RUST_SGX", "1");
            cfg.define("__NO_STRING_INLINES", None);
            cfg.define("__NO_MATH_INLINES", None);
            cfg.define("_LIBUNWIND_IS_BAREMETAL", None);
            cfg.define("__LIBUNWIND_IS_NATIVE_ONLY", None);
            cfg.define("NDEBUG", None);
        } else {
            cfg.flag("-std=c99");
            cfg.flag("-std=c++11");
            cfg.flag("-nostdinc++");
            cfg.flag("-fno-exceptions");
            cfg.flag("-fno-rtti");
            cfg.flag("-fstrict-aliasing");
            cfg.flag("-funwind-tables");
            cfg.flag("-fvisibility=hidden");
            cfg.flag_if_supported("-fvisibility-global-new-delete-hidden");
            cfg.define("_LIBUNWIND_DISABLE_VISIBILITY_ANNOTATIONS", None);
        }

        let mut unwind_sources = vec![
            "Unwind-EHABI.cpp",
            "Unwind-seh.cpp",
            "Unwind-sjlj.c",
            "UnwindLevel1-gcc-ext.c",
            "UnwindLevel1.c",
            "UnwindRegistersRestore.S",
            "UnwindRegistersSave.S",
            "libunwind.cpp",
        ];

        if target_vendor == "apple" {
            unwind_sources.push("Unwind_AppleExtras.cpp");
        }

        if target.contains("x86_64-fortanix-unknown-sgx") {
            unwind_sources.push("UnwindRustSgx.c");
        }

        let root = Path::new("../../src/llvm-project/libunwind");
        cfg.include(root.join("include"));
        for src in unwind_sources {
            cfg.file(root.join("src").join(src));
        }

        if target_env == "musl" {
            // use the same C compiler command to compile C++ code so we do not need to setup the
            // C++ compiler env variables on the builders
            cfg.cpp(false);
            // linking for musl is handled in lib.rs
            cfg.cargo_metadata(false);
            println!("cargo:rustc-link-search=native={}", env::var("OUT_DIR").unwrap());
        }

        cfg.compile("unwind");
    }
}
