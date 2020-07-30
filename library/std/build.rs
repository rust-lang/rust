use std::env;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    let target = env::var("TARGET").expect("TARGET was not set");
    if target.contains("linux") {
        if target.contains("android") {
            println!("cargo:rustc-link-lib=dl");
            println!("cargo:rustc-link-lib=log");
            println!("cargo:rustc-link-lib=gcc");
        } else if !target.contains("musl") {
            println!("cargo:rustc-link-lib=dl");
            println!("cargo:rustc-link-lib=rt");
            println!("cargo:rustc-link-lib=pthread");
        }
    } else if target.contains("freebsd") {
        println!("cargo:rustc-link-lib=execinfo");
        println!("cargo:rustc-link-lib=pthread");
        if env::var("RUST_STD_FREEBSD_12_ABI").is_ok() {
            println!("cargo:rustc-cfg=freebsd12");
        }
    } else if target.contains("netbsd") {
        println!("cargo:rustc-link-lib=pthread");
        println!("cargo:rustc-link-lib=rt");
    } else if target.contains("dragonfly") || target.contains("openbsd") {
        println!("cargo:rustc-link-lib=pthread");
    } else if target.contains("solaris") {
        println!("cargo:rustc-link-lib=socket");
        println!("cargo:rustc-link-lib=posix4");
        println!("cargo:rustc-link-lib=pthread");
        println!("cargo:rustc-link-lib=resolv");
    } else if target.contains("illumos") {
        println!("cargo:rustc-link-lib=socket");
        println!("cargo:rustc-link-lib=posix4");
        println!("cargo:rustc-link-lib=pthread");
        println!("cargo:rustc-link-lib=resolv");
        println!("cargo:rustc-link-lib=nsl");
        // Use libumem for the (malloc-compatible) allocator
        println!("cargo:rustc-link-lib=umem");
    } else if target.contains("apple-darwin") {
        println!("cargo:rustc-link-lib=System");

        // res_init and friends require -lresolv on macOS/iOS.
        // See #41582 and http://blog.achernya.com/2013/03/os-x-has-silly-libsystem.html
        println!("cargo:rustc-link-lib=resolv");
    } else if target.contains("apple-ios") {
        println!("cargo:rustc-link-lib=System");
        println!("cargo:rustc-link-lib=objc");
        println!("cargo:rustc-link-lib=framework=Security");
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=resolv");
    } else if target.contains("uwp") {
        println!("cargo:rustc-link-lib=ws2_32");
        // For BCryptGenRandom
        println!("cargo:rustc-link-lib=bcrypt");
    } else if target.contains("windows") {
        println!("cargo:rustc-link-lib=advapi32");
        println!("cargo:rustc-link-lib=ws2_32");
        println!("cargo:rustc-link-lib=userenv");
    } else if target.contains("fuchsia") {
        println!("cargo:rustc-link-lib=zircon");
        println!("cargo:rustc-link-lib=fdio");
    } else if target.contains("cloudabi") {
        if cfg!(feature = "backtrace") {
            println!("cargo:rustc-link-lib=unwind");
        }
        println!("cargo:rustc-link-lib=c");
        println!("cargo:rustc-link-lib=compiler_rt");
    } else if (target.contains("sgx") && target.contains("fortanix"))
        || target.contains("hermit")
        || target.contains("l4re")
        || target.contains("redox")
        || target.contains("haiku")
        || target.contains("vxworks")
        || target.contains("wasm32")
        || target.contains("asmjs")
    {
        // These platforms don't have any special requirements.
    } else {
        // This is for Cargo's build-std support, to mark std as unstable for
        // typically no_std platforms.
        // This covers:
        // - os=none ("bare metal" targets)
        // - mipsel-sony-psp
        // - nvptx64-nvidia-cuda
        // - avr-unknown-unknown
        // - tvos (aarch64-apple-tvos, x86_64-apple-tvos)
        // - uefi (x86_64-unknown-uefi, i686-unknown-uefi)
        // - JSON targets
        // - Any new targets that have not been explicitly added above.
        println!("cargo:rustc-cfg=feature=\"restricted-std\"");
    }
    println!("cargo:rustc-env=STD_ENV_ARCH={}", env::var("CARGO_CFG_TARGET_ARCH").unwrap());
    println!("cargo:rustc-cfg=backtrace_in_libstd");
}
