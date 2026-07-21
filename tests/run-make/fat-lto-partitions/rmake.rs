// Fat LTO with `-Zfat-lto-partitions` splits the merged module into several
// codegen partitions. Check that a partitioned build produces a working
// binary and that the split is deterministic (two identical invocations
// produce identical binaries).

//@ ignore-cross-compile
//@ needs-target-std

use run_make_support::{rfs, run, rustc};

fn main() {
    for invalid in ["0", "257"] {
        rustc()
            .input("main.rs")
            .arg(format!("-Zfat-lto-partitions={invalid}"))
            .run_fail()
            .assert_stderr_contains("an integer from 1 to 256 was expected");
    }

    let build_single = |snapshot: &str, explicit: bool| {
        let mut cmd = rustc();
        cmd.input("main.rs")
            .crate_name("single")
            .opt_level("3")
            .lto("fat")
            .arg("-Zverify-llvm-ir")
            .output("single-output");
        if explicit {
            cmd.arg("-Zfat-lto-partitions=1");
        }
        cmd.run();
        rfs::copy("single-output", snapshot);
    };
    build_single("single-default", false);
    build_single("single-explicit", true);
    assert_eq!(rfs::read("single-default"), rfs::read("single-explicit"));

    rfs::create_dir("emit");
    rustc()
        .input("main.rs")
        .crate_name("emit_test")
        .opt_level("3")
        .lto("fat")
        .arg("-Zfat-lto-partitions=4")
        .arg("-Zverify-llvm-ir")
        .emit("obj,llvm-bc")
        .out_dir("emit")
        .run();
    let mut object_count = 0;
    let mut bitcode_count = 0;
    for entry in rfs::read_dir("emit") {
        match entry.unwrap().path().extension().and_then(|extension| extension.to_str()) {
            Some("o") => object_count += 1,
            Some("bc") => bitcode_count += 1,
            _ => {}
        }
    }
    assert_eq!(object_count, 5);
    assert_eq!(bitcode_count, 5);

    let build = |out: &str, no_parallel_backend: bool| {
        let mut cmd = rustc();
        cmd.input("main.rs")
            .opt_level("3")
            .lto("fat")
            .arg("-Zfat-lto-partitions=4")
            .arg("-Zverify-llvm-ir")
            .output(out);
        if no_parallel_backend {
            cmd.arg("-Zno-parallel-backend");
        }
        cmd.run();
    };
    build("main-a", false);
    run("main-a");

    build("main-b", false);
    assert_eq!(rfs::read("main-a"), rfs::read("main-b"), "partitioned fat LTO is deterministic");

    build("main-sequential", true);
    assert_eq!(rfs::read("main-a"), rfs::read("main-sequential"));

    check_saturated_jobserver();
}

#[cfg(not(target_os = "linux"))]
fn check_saturated_jobserver() {}

#[cfg(target_os = "linux")]
fn check_saturated_jobserver() {
    use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
    use std::process::Command;
    use std::time::{Duration, Instant};

    use run_make_support::{cwd, env_var, rustc_path};

    // An empty jobserver pipe models a rustc process whose only capacity is the
    // implicit token held by its parent. Partition work must continue on that
    // token instead of waiting forever for another one.
    let mut pipe = [0; 2];
    assert_eq!(unsafe { run_make_support::libc::pipe(pipe.as_mut_ptr()) }, 0);
    let read = unsafe { OwnedFd::from_raw_fd(pipe[0]) };
    let write = unsafe { OwnedFd::from_raw_fd(pipe[1]) };

    let dylib_env = env_var("LD_LIB_PATH_ENVVAR");
    let dylib_path = std::env::join_paths(
        [cwd(), env_var("HOST_RUSTC_DYLIB_PATH").into()]
            .into_iter()
            .chain(std::env::split_paths(&env_var(&dylib_env))),
    )
    .unwrap();
    let jobserver = format!("--jobserver-auth={},{}", read.as_raw_fd(), write.as_raw_fd());
    let mut child = Command::new(rustc_path())
        .args(["main.rs", "-O", "-Clto=fat", "-Zfat-lto-partitions=4", "-o", "jobserver-saturated"])
        .envs(["CARGO_MAKEFLAGS", "MAKEFLAGS", "MFLAGS"].map(|name| (name, &jobserver)))
        .env(dylib_env, dylib_path)
        .spawn()
        .unwrap();

    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        if let Some(status) = child.try_wait().unwrap() {
            assert!(status.success(), "rustc failed with {status}");
            break;
        }
        if Instant::now() >= deadline {
            child.kill().unwrap();
            child.wait().unwrap();
            panic!("partitioned fat LTO stalled with a saturated jobserver");
        }
        std::thread::sleep(Duration::from_millis(20));
    }
}
