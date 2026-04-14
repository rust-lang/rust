//! proc_smoke — ThingOS std::process smoke tests
//!
//! # Child modes (invoked by the test runner as subprocess)
//!
//!   --child-exit <N>   : exit with code N
//!   --child-echo       : print remaining args, one per line, then exit 0
//!   --child-env <KEY>  : print the value of env var KEY, then exit 0
//!   --child-stdin      : read stdin to EOF, echo it to stdout, exit 0
//!
//! # Running
//!
//!   proc_smoke               — run all tests
//!   proc_smoke --list        — list test names
//!
//! Each test prints PASS or FAIL and a description.
#![no_std]
#![no_main]
use alloc::string::ToString;
use core::default::Default;
extern crate alloc;


use std::process::Command;

// ─── child modes ─────────────────────────────────────────────────────────────

fn child_dispatch() -> bool {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        return false;
    }
    match args[1].as_str() {
        "--child-exit" => {
            let code: i32 = if args.len() > 2 {
                args[2].parse().unwrap_or(1)
            } else {
                0
            };
            std::process::exit(code);
        }
        "--child-echo" => {
            for arg in args.iter().skip(2) {
                println!("{}", arg);
            }
            std::process::exit(0);
        }
        "--child-env" => {
            let key = if args.len() > 2 { args[2].as_str() } else { "" };
            match std::env::var(key) {
                Ok(val) => println!("{}", val),
                Err(_) => println!("<unset>"),
            }
            std::process::exit(0);
        }
        "--child-stdin" => {
            use std::io::{self, Read, Write};
            let mut buf = Vec::new();
            io::stdin().read_to_end(&mut buf).unwrap();
            io::stdout().write_all(&buf).unwrap();
            std::process::exit(0);
        }
        "--child-deadlock" => {
            // Use direct syscalls to avoid std::io dependencies in this mode.
            // Write 128KB to stderr (fd 2).
            let data = [0u8; 1024];
            for _ in 0..128 {
                let _ = abi::syscall::vfs_write(2, &data);
            }
            // Signal completion on stdout (fd 1).
            let _ = abi::syscall::vfs_write(1, b"done");
            std::process::exit(0);
        }
        _ => return false,
    }
}

// ─── test harness ────────────────────────────────────────────────────────────

struct Test {
    name: &'static str,
    run: fn(&str) -> Result<(), String>,
}

fn self_exe() -> String {
    std::env::args().next().unwrap_or_else(|| "/proc_smoke".to_string())
}

// ─── individual tests ─────────────────────────────────────────────────────────

/// proc_status: spawn a child that exits 0; verify success().
fn test_proc_status(exe: &str) -> Result<(), String> {
    let status = Command::new(exe)
        .arg("--child-exit")
        .arg("0")
        .status()
        .map_err(|e| alloc::format!("spawn failed: {}", e))?;
    if !status.success() {
        return Err(alloc::format!("expected success, got {:?}", status.code()));
    }
    Ok(())
}

/// proc_exit_code: spawn a child that exits 7; verify code() == Some(7).
fn test_proc_exit_code(exe: &str) -> Result<(), String> {
    let status = Command::new(exe)
        .arg("--child-exit")
        .arg("7")
        .status()
        .map_err(|e| alloc::format!("spawn failed: {}", e))?;
    match status.code() {
        Some(7) => Ok(()),
        other => Err(alloc::format!("expected code 7, got {:?}", other)),
    }
}

/// proc_args: spawn echo child; verify args are passed correctly.
fn test_proc_args(exe: &str) -> Result<(), String> {
    let output = Command::new(exe)
        .arg("--child-echo")
        .arg("hello")
        .arg("world")
        .output()
        .map_err(|e| alloc::format!("spawn failed: {}", e))?;
    if !output.status.success() {
        return Err(alloc::format!("child exited {:?}", output.status.code()));
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    if !stdout.contains("hello") {
        return Err(alloc::format!("missing 'hello' in output: {:?}", stdout));
    }
    if !stdout.contains("world") {
        return Err(alloc::format!("missing 'world' in output: {:?}", stdout));
    }
    Ok(())
}

/// proc_env: inject an env var; verify child sees it.
fn test_proc_env(exe: &str) -> Result<(), String> {
    let output = Command::new(exe)
        .arg("--child-env")
        .arg("SMOKE_TEST_VAR")
        .env("SMOKE_TEST_VAR", "thingos_rocks")
        .output()
        .map_err(|e| alloc::format!("spawn failed: {}", e))?;
    if !output.status.success() {
        return Err(alloc::format!("child exited {:?}", output.status.code()));
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    if !stdout.trim().contains("thingos_rocks") {
        return Err(alloc::format!("expected 'thingos_rocks' in output, got: {:?}", stdout));
    }
    Ok(())
}

/// proc_stdout_pipe: capture child stdout via .output().
fn test_proc_stdout_pipe(exe: &str) -> Result<(), String> {
    let output = Command::new(exe)
        .arg("--child-echo")
        .arg("piped_output")
        .output()
        .map_err(|e| alloc::format!("spawn failed: {}", e))?;
    if !output.status.success() {
        return Err(alloc::format!("child exited {:?}", output.status.code()));
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    if !stdout.contains("piped_output") {
        return Err(alloc::format!("expected 'piped_output', got: {:?}", stdout));
    }
    Ok(())
}

/// proc_stdin_pipe: write to child stdin; verify echoed output.
fn test_proc_stdin_pipe(exe: &str) -> Result<(), String> {
    use std::io::Write;
    use std::process::Stdio;

    let mut child = Command::new(exe)
        .arg("--child-stdin")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .map_err(|e| alloc::format!("spawn failed: {}", e))?;

    // Write to child stdin then close it so the child sees EOF.
    {
        let stdin = child.stdin.take().ok_or("no stdin pipe")?;
        let mut stdin = stdin;
        stdin.write_all(b"hello from parent").map_err(|e| alloc::format!("write: {}", e))?;
        // Drop to close the write end, signalling EOF to the child.
    }

    let output = child.wait_with_output().map_err(|e| alloc::format!("wait: {}", e))?;
    if !output.status.success() {
        return Err(alloc::format!("child exited {:?}", output.status.code()));
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    if !stdout.contains("hello from parent") {
        return Err(alloc::format!("expected echo, got: {:?}", stdout));
    }
    Ok(())
}

/// proc_inherit_stdio: spawn child with inherited stdio (output goes to console).
fn test_proc_inherit_stdio(exe: &str) -> Result<(), String> {
    let status = Command::new(exe)
        .arg("--child-echo")
        .arg("inherited_stdio_ok")
        .status()
        .map_err(|e| alloc::format!("spawn failed: {}", e))?;
    if !status.success() {
        return Err(alloc::format!("child exited {:?}", status.code()));
    }
    // Output appears on the console naturally — no capture needed.
    Ok(())
}

/// proc_try_wait: spawn a process; poll with try_wait until done.
fn test_proc_try_wait(exe: &str) -> Result<(), String> {
    let mut child = Command::new(exe)
        .arg("--child-exit")
        .arg("3")
        .spawn()
        .map_err(|e| alloc::format!("spawn failed: {}", e))?;

    // Poll a few times then fall back to blocking wait.
    let status = loop {
        match child.try_wait().map_err(|e| alloc::format!("try_wait: {}", e))? {
            Some(s) => break s,
            None => {
                // Child still running; yield and retry.
                std::thread::yield_now();
            }
        }
    };
    match status.code() {
        Some(3) => Ok(()),
        other => Err(alloc::format!("expected code 3, got {:?}", other)),
    }
}

/// proc_deadlock: spawn child that fills stderr before writing stdout; verify no hang.
fn test_proc_deadlock(exe: &str) -> Result<(), String> {
    let output = Command::new(exe)
        .arg("--child-deadlock")
        .output()
        .map_err(|e| alloc::format!("spawn failed: {}", e))?;

    if !output.status.success() {
        return Err(alloc::format!("child exited {:?}", output.status.code()));
    }
    if output.stderr.len() < 64 * 1024 {
        return Err(alloc::format!("expected large stderr, got {}", output.stderr.len()));
    }
    if !String::from_utf8_lossy(&output.stdout).contains("done") {
        return Err(alloc::format!("missing 'done' in stdout, got {:?}", output.stdout));
    }
    Ok(())
}

/// proc_pipeline: spawn child 1 (echo "hello") -> pipe -> child 2 (cat); verify output.
fn test_proc_pipeline(exe: &str) -> Result<(), String> {
    use std::process::Stdio;

    // child 1: echo "pipeline_data"
    let mut child1 = Command::new(exe)
        .arg("--child-echo")
        .arg("pipeline_data")
        .stdout(Stdio::piped())
        .spawn()
        .map_err(|e| alloc::format!("spawn child1 failed: {}", e))?;

    let stdout1 = child1.stdout.take().ok_or("no child1 stdout")?;

    // child 2: cat (echoes stdin)
    // Here we pass ChildStdout (ChildPipe) directly to stdin.
    // This relies on the new FdRemap logic in Command::spawn.
    let output2 = Command::new(exe)
        .arg("--child-stdin")
        .stdin(stdout1)
        .output()
        .map_err(|e| alloc::format!("spawn child2 failed: {}", e))?;

    if !output2.status.success() {
        return Err(alloc::format!("child2 exited {:?}", output2.status.code()));
    }
    let stdout2 = String::from_utf8_lossy(&output2.stdout);
    if !stdout2.contains("pipeline_data") {
        return Err(alloc::format!("expected 'pipeline_data' in output2, got: {:?}", stdout2));
    }

    // Cleanup child1
    let status1 = child1.wait().map_err(|e| alloc::format!("wait child1 failed: {}", e))?;
    if !status1.success() {
        return Err(alloc::format!("child1 exited with failure: {:?}", status1.code()));
    }

    Ok(())
}

// ─── main ─────────────────────────────────────────────────────────────────────

fn main() {
    // If we were invoked as a child process, handle that and never return.
    if child_dispatch() {
        unreachable!();
    }

    let exe = self_exe();

    let tests: &[Test] = &[
        Test { name: "proc_status",        run: test_proc_status },
        Test { name: "proc_exit_code",     run: test_proc_exit_code },
        Test { name: "proc_args",          run: test_proc_args },
        Test { name: "proc_env",           run: test_proc_env },
        Test { name: "proc_stdout_pipe",   run: test_proc_stdout_pipe },
        Test { name: "proc_stdin_pipe",    run: test_proc_stdin_pipe },
        Test { name: "proc_inherit_stdio", run: test_proc_inherit_stdio },
        Test { name: "proc_try_wait",      run: test_proc_try_wait },
        Test { name: "proc_deadlock",      run: test_proc_deadlock },
        Test { name: "proc_pipeline",      run: test_proc_pipeline },
    ];

    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--list") {
        for t in tests {
            println!("{}", t.name);
        }
        return;
    }

    let mut passed = 0usize;
    let mut failed = 0usize;

    for t in tests {
        match (t.run)(&exe) {
            Ok(()) => {
                println!("PASS  {}", t.name);
                passed += 1;
            }
            Err(msg) => {
                eprintln!("FAIL  {}  — {}", t.name, msg);
                failed += 1;
            }
        }
    }

    println!("\n{} passed, {} failed", passed, failed);
    std::process::exit(if failed == 0 { 0 } else { 1 });
}
