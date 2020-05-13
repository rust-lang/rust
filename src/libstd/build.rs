use std::env;
use std::path::PathBuf;
use std::process::Command;

const BOEHM_REPO: &str = "https://github.com/ivmai/bdwgc.git";
const BOEHM_ATOMICS_REPO: &str = "https://github.com/ivmai/libatomic_ops.git";
const BOEHM_DIR: &str = "bdwgc";
const BUILD_DIR: &str = ".libs";

#[cfg(not(all(target_pointer_width = "64", target_arch = "x86_64")))]
compile_error!("Requires x86_64 with 64 bit pointer width.");
static POINTER_MASK: &str = "-DPOINTER_MASK=0xFFFFFFFFFFFFFFF8";
static FPIC: &str = "-fPIC";

fn run<F>(name: &str, mut configure: F)
where
    F: FnMut(&mut Command) -> &mut Command,
{
    let mut command = Command::new(name);
    let configured = configure(&mut command);
    if !configured.status().is_ok() {
        panic!("failed to execute {:?}", configured);
    }
}

fn main() {
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
    }


    let out_dir = env::var("OUT_DIR").unwrap();
    let mut boehm_src = PathBuf::from(out_dir);
    boehm_src.push(BOEHM_DIR);

    if !boehm_src.exists() {
        run("git", |cmd| {
            cmd.arg("clone").arg(BOEHM_REPO).arg(&boehm_src)
        });

        run("git", |cmd| {
            cmd.arg("clone")
                .arg(BOEHM_ATOMICS_REPO)
                .current_dir(&boehm_src)
        });
    }

    env::set_current_dir(&boehm_src).unwrap();

    run("./autogen.sh", |cmd| cmd);
    run("./configure", |cmd| {
        cmd.arg("--enable-static")
            .arg("--disable-shared")
            .env("CFLAGS", format!("{} {}", POINTER_MASK, FPIC))
    });

    run("make", |cmd| cmd );

    let mut libpath = PathBuf::from(&boehm_src);
    libpath.push(BUILD_DIR);

    println!(
        "cargo:rustc-link-search=native={}",
        &libpath.as_path().to_str().unwrap()
    );
    println!("cargo:rustc-link-lib=static=gc");
}
