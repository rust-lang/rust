#![no_std]
#![no_main]
use alloc::string::ToString;
use core::default::Default;
extern crate alloc;

use alloc::collections::BTreeMap;
use stem::println;
use stem::syscall::{execv, execve, getpid, vfs_close, vfs_mkdir, vfs_open, vfs_unlink, vfs_write};

/// Write `content` to `path`, creating/overwriting it.  Returns `Ok(())` on
/// success or a string describing the failure.
fn write_file(path: &str, content: &[u8]) -> Result<(), &'static str> {
    let fd = vfs_open(
        path,
        abi::syscall::vfs_flags::O_RDWR | abi::syscall::vfs_flags::O_CREAT,
    )
    .map_err(|_| "open failed")?;
    let mut written = 0;
    while written < content.len() {
        match vfs_write(fd, &content[written..]) {
            Ok(n) if n > 0 => written += n,
            _ => {
                let _ = vfs_close(fd);
                return Err("write failed");
            }
        }
    }
    let _ = vfs_close(fd);
    Ok(())
}

#[stem::main]
fn main(_arg0: usize) -> ! {
    let pid = getpid();
    println!("TEST_EXEC: Starting. PID={}", pid);

    // ── Test 1: execve with a non-existent path should return ENOENT ─────────
    {
        let res = execve("/does/not/exist", &[], &BTreeMap::new());
        match res {
            Err(e) => println!("TEST_EXEC: [PASS] non-existent path returned error: {:?}", e),
            Ok(()) => {
                println!("TEST_EXEC: [FAIL] execve of non-existent path succeeded unexpectedly");
                stem::syscall::exit(-1);
            }
        }
    }

    // ── Test 2: execv with a non-existent path should return ENOENT ──────────
    {
        let res = execv("/does/not/exist", &[b"/does/not/exist"]);
        match res {
            Err(e) => println!("TEST_EXEC: [PASS] execv non-existent path returned error: {:?}", e),
            Ok(()) => {
                println!("TEST_EXEC: [FAIL] execv of non-existent path succeeded unexpectedly");
                stem::syscall::exit(-1);
            }
        }
    }

    // ── Test 3: shebang with non-existent interpreter returns an error ────────
    {
        let _ = vfs_mkdir("/tmp");
        let script_path = "/tmp/test_shebang_bad.sh";
        let script_content = b"#!/does/not/exist/interpreter\necho hello\n";
        match write_file(script_path, script_content) {
            Ok(()) => {
                let res = execve(script_path, &[b"test_shebang_bad.sh"], &BTreeMap::new());
                match res {
                    Err(e) => println!(
                        "TEST_EXEC: [PASS] shebang with missing interpreter returned error: {:?}",
                        e
                    ),
                    Ok(()) => {
                        println!("TEST_EXEC: [FAIL] shebang with missing interpreter succeeded unexpectedly");
                        stem::syscall::exit(-1);
                    }
                }
                let _ = vfs_unlink(script_path);
            }
            Err(reason) => {
                println!(
                    "TEST_EXEC: [SKIP] could not create shebang test script ({}), skipping",
                    reason
                );
            }
        }
    }

    // ── Test 4: non-executable file (no ELF/shebang magic) returns ENOEXEC ───
    {
        let bad_path = "/tmp/test_not_exec.txt";
        match write_file(bad_path, b"this is just a text file\n") {
            Ok(()) => {
                let res = execve(bad_path, &[b"test_not_exec.txt"], &BTreeMap::new());
                match res {
                    Err(e) => println!(
                        "TEST_EXEC: [PASS] non-executable file returned error: {:?}",
                        e
                    ),
                    Ok(()) => {
                        println!(
                            "TEST_EXEC: [FAIL] non-executable file exec succeeded unexpectedly"
                        );
                        stem::syscall::exit(-1);
                    }
                }
                let _ = vfs_unlink(bad_path);
            }
            Err(reason) => {
                println!(
                    "TEST_EXEC: [SKIP] could not create non-exec test file ({}), skipping",
                    reason
                );
            }
        }
    }

    // ── Test 5: execve a valid ELF binary (replaces this process) ────────────
    // We assume /bin/echo exists on the system.
    let path = "/bin/echo";
    let args: &[&[u8]] = &[
        b"echo", b"Hello", b"from", b"execve!", b"(PID", b"should", b"be", b"the", b"same)",
    ];
    let env = BTreeMap::new();

    println!("TEST_EXEC: Executing {} with args...", path);
    let res = execve(path, args, &env);

    // If we reach here, execve failed.
    println!("TEST_EXEC: execve FAILED with error: {:?}", res);
    stem::syscall::exit(-1);
}
