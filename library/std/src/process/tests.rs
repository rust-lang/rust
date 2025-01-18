use super::{Command, Output, Stdio};
use crate::io::prelude::*;
use crate::io::{BorrowedBuf, ErrorKind};
use crate::mem::MaybeUninit;
use crate::str;

fn known_command() -> Command {
    if cfg!(windows) { Command::new("help") } else { Command::new("echo") }
}

#[cfg(target_os = "android")]
fn shell_cmd() -> Command {
    Command::new("/system/bin/sh")
}

#[cfg(not(target_os = "android"))]
fn shell_cmd() -> Command {
    Command::new("/bin/sh")
}

#[test]
#[cfg_attr(any(target_os = "vxworks"), ignore)]
fn smoke() {
    let p = if cfg!(target_os = "windows") {
        Command::new("cmd").args(&["/C", "exit 0"]).spawn()
    } else {
        shell_cmd().arg("-c").arg("true").spawn()
    };
    assert!(p.is_ok());
    let mut p = p.unwrap();
    assert!(p.wait().unwrap().success());
}

#[test]
#[cfg_attr(target_os = "android", ignore)]
fn smoke_failure() {
    match Command::new("if-this-is-a-binary-then-the-world-has-ended").spawn() {
        Ok(..) => panic!(),
        Err(..) => {}
    }
}

#[test]
#[cfg_attr(any(target_os = "vxworks"), ignore)]
fn exit_reported_right() {
    let p = if cfg!(target_os = "windows") {
        Command::new("cmd").args(&["/C", "exit 1"]).spawn()
    } else {
        shell_cmd().arg("-c").arg("false").spawn()
    };
    assert!(p.is_ok());
    let mut p = p.unwrap();
    assert!(p.wait().unwrap().code() == Some(1));
    drop(p.wait());
}

#[test]
#[cfg(unix)]
#[cfg_attr(any(target_os = "vxworks"), ignore)]
fn signal_reported_right() {
    use crate::os::unix::process::ExitStatusExt;

    let mut p = shell_cmd().arg("-c").arg("read a").stdin(Stdio::piped()).spawn().unwrap();
    p.kill().unwrap();
    match p.wait().unwrap().signal() {
        Some(9) => {}
        result => panic!("not terminated by signal 9 (instead, {result:?})"),
    }
}

pub fn run_output(mut cmd: Command) -> String {
    let p = cmd.spawn();
    assert!(p.is_ok());
    let mut p = p.unwrap();
    assert!(p.stdout.is_some());
    let mut ret = String::new();
    p.stdout.as_mut().unwrap().read_to_string(&mut ret).unwrap();
    assert!(p.wait().unwrap().success());
    return ret;
}

#[test]
#[cfg_attr(any(target_os = "vxworks"), ignore)]
fn stdout_works() {
    if cfg!(target_os = "windows") {
        let mut cmd = Command::new("cmd");
        cmd.args(&["/C", "echo foobar"]).stdout(Stdio::piped());
        assert_eq!(run_output(cmd), "foobar\r\n");
    } else {
        let mut cmd = shell_cmd();
        cmd.arg("-c").arg("echo foobar").stdout(Stdio::piped());
        assert_eq!(run_output(cmd), "foobar\n");
    }
}

#[test]
#[cfg_attr(any(windows, target_os = "vxworks"), ignore)]
fn set_current_dir_works() {
    // On many Unix platforms this will use the posix_spawn path.
    let mut cmd = shell_cmd();
    cmd.arg("-c").arg("pwd").current_dir("/").stdout(Stdio::piped());
    assert_eq!(run_output(cmd), "/\n");

    // Also test the fork/exec path by setting a pre_exec function.
    #[cfg(unix)]
    {
        use crate::os::unix::process::CommandExt;

        let mut cmd = shell_cmd();
        cmd.arg("-c").arg("pwd").current_dir("/").stdout(Stdio::piped());
        unsafe {
            cmd.pre_exec(|| Ok(()));
        }
        assert_eq!(run_output(cmd), "/\n");
    }
}

#[test]
#[cfg_attr(any(windows, target_os = "vxworks"), ignore)]
fn stdin_works() {
    let mut p = shell_cmd()
        .arg("-c")
        .arg("read line; echo $line")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    p.stdin.as_mut().unwrap().write("foobar".as_bytes()).unwrap();
    drop(p.stdin.take());
    let mut out = String::new();
    p.stdout.as_mut().unwrap().read_to_string(&mut out).unwrap();
    assert!(p.wait().unwrap().success());
    assert_eq!(out, "foobar\n");
}

#[test]
#[cfg_attr(any(target_os = "vxworks"), ignore)]
fn child_stdout_read_buf() {
    let mut cmd = if cfg!(target_os = "windows") {
        let mut cmd = Command::new("cmd");
        cmd.arg("/C").arg("echo abc");
        cmd
    } else {
        let mut cmd = shell_cmd();
        cmd.arg("-c").arg("echo abc");
        cmd
    };
    cmd.stdin(Stdio::null());
    cmd.stdout(Stdio::piped());
    let child = cmd.spawn().unwrap();

    let mut stdout = child.stdout.unwrap();
    let mut buf: [MaybeUninit<u8>; 128] = [MaybeUninit::uninit(); 128];
    let mut buf = BorrowedBuf::from(buf.as_mut_slice());
    stdout.read_buf(buf.unfilled()).unwrap();

    // ChildStdout::read_buf should omit buffer initialization.
    if cfg!(target_os = "windows") {
        assert_eq!(buf.filled(), b"abc\r\n");
        assert_eq!(buf.init_len(), 5);
    } else {
        assert_eq!(buf.filled(), b"abc\n");
        assert_eq!(buf.init_len(), 4);
    };
}

#[test]
#[cfg_attr(any(target_os = "vxworks"), ignore)]
fn test_process_status() {
    let mut status = if cfg!(target_os = "windows") {
        Command::new("cmd").args(&["/C", "exit 1"]).status().unwrap()
    } else {
        shell_cmd().arg("-c").arg("false").status().unwrap()
    };
    assert!(status.code() == Some(1));

    status = if cfg!(target_os = "windows") {
        Command::new("cmd").args(&["/C", "exit 0"]).status().unwrap()
    } else {
        shell_cmd().arg("-c").arg("true").status().unwrap()
    };
    assert!(status.success());
}

#[test]
fn test_process_output_fail_to_start() {
    match Command::new("/no-binary-by-this-name-should-exist").output() {
        Err(e) => assert_eq!(e.kind(), ErrorKind::NotFound),
        Ok(..) => panic!(),
    }
}

#[test]
#[cfg_attr(any(target_os = "vxworks"), ignore)]
fn test_process_output_output() {
    let Output { status, stdout, stderr } = if cfg!(target_os = "windows") {
        Command::new("cmd").args(&["/C", "echo hello"]).output().unwrap()
    } else {
        shell_cmd().arg("-c").arg("echo hello").output().unwrap()
    };
    let output_str = str::from_utf8(&stdout).unwrap();

    assert!(status.success());
    assert_eq!(output_str.trim().to_string(), "hello");
    assert_eq!(stderr, Vec::new());
}

#[test]
#[cfg_attr(any(target_os = "vxworks"), ignore)]
fn test_process_output_error() {
    let Output { status, stdout, stderr } = if cfg!(target_os = "windows") {
        Command::new("cmd").args(&["/C", "mkdir ."]).output().unwrap()
    } else {
        Command::new("mkdir").arg("./").output().unwrap()
    };

    assert!(status.code().is_some());
    assert!(status.code() != Some(0));
    assert_eq!(stdout, Vec::new());
    assert!(!stderr.is_empty());
}

#[test]
#[cfg_attr(any(target_os = "vxworks"), ignore)]
fn test_finish_once() {
    let mut prog = if cfg!(target_os = "windows") {
        Command::new("cmd").args(&["/C", "exit 1"]).spawn().unwrap()
    } else {
        shell_cmd().arg("-c").arg("false").spawn().unwrap()
    };
    assert!(prog.wait().unwrap().code() == Some(1));
}

#[test]
#[cfg_attr(any(target_os = "vxworks"), ignore)]
fn test_finish_twice() {
    let mut prog = if cfg!(target_os = "windows") {
        Command::new("cmd").args(&["/C", "exit 1"]).spawn().unwrap()
    } else {
        shell_cmd().arg("-c").arg("false").spawn().unwrap()
    };
    assert!(prog.wait().unwrap().code() == Some(1));
    assert!(prog.wait().unwrap().code() == Some(1));
}

#[test]
#[cfg_attr(any(target_os = "vxworks"), ignore)]
fn test_wait_with_output_once() {
    let prog = if cfg!(target_os = "windows") {
        Command::new("cmd").args(&["/C", "echo hello"]).stdout(Stdio::piped()).spawn().unwrap()
    } else {
        shell_cmd().arg("-c").arg("echo hello").stdout(Stdio::piped()).spawn().unwrap()
    };

    let Output { status, stdout, stderr } = prog.wait_with_output().unwrap();
    let output_str = str::from_utf8(&stdout).unwrap();

    assert!(status.success());
    assert_eq!(output_str.trim().to_string(), "hello");
    assert_eq!(stderr, Vec::new());
}

#[cfg(all(unix, not(target_os = "android")))]
pub fn env_cmd() -> Command {
    Command::new("env")
}
#[cfg(target_os = "android")]
pub fn env_cmd() -> Command {
    let mut cmd = Command::new("/system/bin/sh");
    cmd.arg("-c").arg("set");
    cmd
}

#[cfg(windows)]
pub fn env_cmd() -> Command {
    let mut cmd = Command::new("cmd");
    cmd.arg("/c").arg("set");
    cmd
}

#[test]
#[cfg_attr(target_os = "vxworks", ignore)]
fn test_override_env() {
    use crate::env;

    // In some build environments (such as chrooted Nix builds), `env` can
    // only be found in the explicitly-provided PATH env variable, not in
    // default places such as /bin or /usr/bin. So we need to pass through
    // PATH to our sub-process.
    let mut cmd = env_cmd();
    cmd.env_clear().env("RUN_TEST_NEW_ENV", "123");
    if let Some(p) = env::var_os("PATH") {
        cmd.env("PATH", &p);
    }
    let result = cmd.output().unwrap();
    let output = String::from_utf8_lossy(&result.stdout).to_string();

    assert!(
        output.contains("RUN_TEST_NEW_ENV=123"),
        "didn't find RUN_TEST_NEW_ENV inside of:\n\n{output}",
    );
}

#[test]
#[cfg_attr(target_os = "vxworks", ignore)]
fn test_add_to_env() {
    let result = env_cmd().env("RUN_TEST_NEW_ENV", "123").output().unwrap();
    let output = String::from_utf8_lossy(&result.stdout).to_string();

    assert!(
        output.contains("RUN_TEST_NEW_ENV=123"),
        "didn't find RUN_TEST_NEW_ENV inside of:\n\n{output}"
    );
}

#[test]
#[cfg_attr(target_os = "vxworks", ignore)]
fn test_capture_env_at_spawn() {
    use crate::env;

    let mut cmd = env_cmd();
    cmd.env("RUN_TEST_NEW_ENV1", "123");

    // This variable will not be present if the environment has already
    // been captured above.
    env::set_var("RUN_TEST_NEW_ENV2", "456");
    let result = cmd.output().unwrap();
    env::remove_var("RUN_TEST_NEW_ENV2");

    let output = String::from_utf8_lossy(&result.stdout).to_string();

    assert!(
        output.contains("RUN_TEST_NEW_ENV1=123"),
        "didn't find RUN_TEST_NEW_ENV1 inside of:\n\n{output}"
    );
    assert!(
        output.contains("RUN_TEST_NEW_ENV2=456"),
        "didn't find RUN_TEST_NEW_ENV2 inside of:\n\n{output}"
    );
}

// Regression tests for #30858.
#[test]
fn test_interior_nul_in_progname_is_error() {
    match Command::new("has-some-\0\0s-inside").spawn() {
        Err(e) => assert_eq!(e.kind(), ErrorKind::InvalidInput),
        Ok(_) => panic!(),
    }
}

#[test]
fn test_interior_nul_in_arg_is_error() {
    match known_command().arg("has-some-\0\0s-inside").spawn() {
        Err(e) => assert_eq!(e.kind(), ErrorKind::InvalidInput),
        Ok(_) => panic!(),
    }
}

#[test]
fn test_interior_nul_in_args_is_error() {
    match known_command().args(&["has-some-\0\0s-inside"]).spawn() {
        Err(e) => assert_eq!(e.kind(), ErrorKind::InvalidInput),
        Ok(_) => panic!(),
    }
}

#[test]
fn test_interior_nul_in_current_dir_is_error() {
    match known_command().current_dir("has-some-\0\0s-inside").spawn() {
        Err(e) => assert_eq!(e.kind(), ErrorKind::InvalidInput),
        Ok(_) => panic!(),
    }
}

// Regression tests for #30862.
#[test]
#[cfg_attr(target_os = "vxworks", ignore)]
fn test_interior_nul_in_env_key_is_error() {
    match env_cmd().env("has-some-\0\0s-inside", "value").spawn() {
        Err(e) => assert_eq!(e.kind(), ErrorKind::InvalidInput),
        Ok(_) => panic!(),
    }
}

#[test]
#[cfg_attr(target_os = "vxworks", ignore)]
fn test_interior_nul_in_env_value_is_error() {
    match env_cmd().env("key", "has-some-\0\0s-inside").spawn() {
        Err(e) => assert_eq!(e.kind(), ErrorKind::InvalidInput),
        Ok(_) => panic!(),
    }
}

/// Tests that process creation flags work by debugging a process.
/// Other creation flags make it hard or impossible to detect
/// behavioral changes in the process.
#[test]
#[cfg(windows)]
fn test_creation_flags() {
    use crate::os::windows::process::CommandExt;
    use crate::sys::c::{BOOL, INFINITE};
    #[repr(C)]
    struct DEBUG_EVENT {
        pub event_code: u32,
        pub process_id: u32,
        pub thread_id: u32,
        // This is a union in the real struct, but we don't
        // need this data for the purposes of this test.
        pub _junk: [u8; 164],
    }

    extern "system" {
        fn WaitForDebugEvent(lpDebugEvent: *mut DEBUG_EVENT, dwMilliseconds: u32) -> BOOL;
        fn ContinueDebugEvent(dwProcessId: u32, dwThreadId: u32, dwContinueStatus: u32) -> BOOL;
    }

    const DEBUG_PROCESS: u32 = 1;
    const EXIT_PROCESS_DEBUG_EVENT: u32 = 5;
    const DBG_EXCEPTION_NOT_HANDLED: u32 = 0x80010001;

    let mut child = Command::new("cmd")
        .creation_flags(DEBUG_PROCESS)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(b"exit\r\n").unwrap();
    let mut events = 0;
    let mut event = DEBUG_EVENT { event_code: 0, process_id: 0, thread_id: 0, _junk: [0; 164] };
    loop {
        if unsafe { WaitForDebugEvent(&mut event as *mut DEBUG_EVENT, INFINITE) } == 0 {
            panic!("WaitForDebugEvent failed!");
        }
        events += 1;

        if event.event_code == EXIT_PROCESS_DEBUG_EVENT {
            break;
        }

        if unsafe {
            ContinueDebugEvent(event.process_id, event.thread_id, DBG_EXCEPTION_NOT_HANDLED)
        } == 0
        {
            panic!("ContinueDebugEvent failed!");
        }
    }
    assert!(events > 0);
}

/// Tests proc thread attributes by spawning a process with a custom parent process,
/// then comparing the parent process ID with the expected parent process ID.
#[test]
#[cfg(windows)]
fn test_proc_thread_attributes() {
    use crate::mem;
    use crate::os::windows::io::AsRawHandle;
    use crate::os::windows::process::{CommandExt, ProcThreadAttributeList};
    use crate::sys::c::{BOOL, CloseHandle, HANDLE};
    use crate::sys::cvt;

    #[repr(C)]
    #[allow(non_snake_case)]
    struct PROCESSENTRY32W {
        dwSize: u32,
        cntUsage: u32,
        th32ProcessID: u32,
        th32DefaultHeapID: usize,
        th32ModuleID: u32,
        cntThreads: u32,
        th32ParentProcessID: u32,
        pcPriClassBase: i32,
        dwFlags: u32,
        szExeFile: [u16; 260],
    }

    extern "system" {
        fn CreateToolhelp32Snapshot(dwflags: u32, th32processid: u32) -> HANDLE;
        fn Process32First(hsnapshot: HANDLE, lppe: *mut PROCESSENTRY32W) -> BOOL;
        fn Process32Next(hsnapshot: HANDLE, lppe: *mut PROCESSENTRY32W) -> BOOL;
    }

    const PROC_THREAD_ATTRIBUTE_PARENT_PROCESS: usize = 0x00020000;
    const TH32CS_SNAPPROCESS: u32 = 0x00000002;

    struct ProcessDropGuard(crate::process::Child);

    impl Drop for ProcessDropGuard {
        fn drop(&mut self) {
            let _ = self.0.kill();
        }
    }

    let mut parent = Command::new("cmd");
    parent.stdout(Stdio::null()).stderr(Stdio::null());

    let parent = ProcessDropGuard(parent.spawn().unwrap());

    let mut child_cmd = Command::new("cmd");
    child_cmd.stdout(Stdio::null()).stderr(Stdio::null());

    let parent_process_handle = parent.0.as_raw_handle();

    let mut attribute_list = ProcThreadAttributeList::build()
        .attribute(PROC_THREAD_ATTRIBUTE_PARENT_PROCESS, &parent_process_handle)
        .finish()
        .unwrap();

    let child = ProcessDropGuard(child_cmd.spawn_with_attributes(&mut attribute_list).unwrap());

    let h_snapshot = unsafe { CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0) };

    let mut process_entry = PROCESSENTRY32W {
        dwSize: mem::size_of::<PROCESSENTRY32W>() as u32,
        cntUsage: 0,
        th32ProcessID: 0,
        th32DefaultHeapID: 0,
        th32ModuleID: 0,
        cntThreads: 0,
        th32ParentProcessID: 0,
        pcPriClassBase: 0,
        dwFlags: 0,
        szExeFile: [0; 260],
    };

    unsafe { cvt(Process32First(h_snapshot, &mut process_entry as *mut _)) }.unwrap();

    loop {
        if child.0.id() == process_entry.th32ProcessID {
            break;
        }
        unsafe { cvt(Process32Next(h_snapshot, &mut process_entry as *mut _)) }.unwrap();
    }

    unsafe { cvt(CloseHandle(h_snapshot)) }.unwrap();

    assert_eq!(parent.0.id(), process_entry.th32ParentProcessID);

    drop(child)
}

#[test]
fn test_command_implements_send_sync() {
    fn take_send_sync_type<T: Send + Sync>(_: T) {}
    take_send_sync_type(Command::new(""))
}

// Ensure that starting a process with no environment variables works on Windows.
// This will fail if the environment block is ill-formed.
#[test]
#[cfg(windows)]
fn env_empty() {
    let p = Command::new("cmd").args(&["/C", "exit 0"]).env_clear().spawn();
    assert!(p.is_ok());
}

#[test]
#[cfg(not(windows))]
#[cfg_attr(any(target_os = "emscripten", target_env = "sgx"), ignore)]
fn debug_print() {
    const PIDFD: &'static str =
        if cfg!(target_os = "linux") { "    create_pidfd: false,\n" } else { "" };

    let mut command = Command::new("some-boring-name");

    assert_eq!(format!("{command:?}"), format!(r#""some-boring-name""#));

    assert_eq!(
        format!("{command:#?}"),
        format!(
            r#"Command {{
    program: "some-boring-name",
    args: [
        "some-boring-name",
    ],
{PIDFD}}}"#
        )
    );

    command.args(&["1", "2", "3"]);

    assert_eq!(format!("{command:?}"), format!(r#""some-boring-name" "1" "2" "3""#));

    assert_eq!(
        format!("{command:#?}"),
        format!(
            r#"Command {{
    program: "some-boring-name",
    args: [
        "some-boring-name",
        "1",
        "2",
        "3",
    ],
{PIDFD}}}"#
        )
    );

    crate::os::unix::process::CommandExt::arg0(&mut command, "exciting-name");

    assert_eq!(
        format!("{command:?}"),
        format!(r#"["some-boring-name"] "exciting-name" "1" "2" "3""#)
    );

    assert_eq!(
        format!("{command:#?}"),
        format!(
            r#"Command {{
    program: "some-boring-name",
    args: [
        "exciting-name",
        "1",
        "2",
        "3",
    ],
{PIDFD}}}"#
        )
    );

    let mut command_with_env_and_cwd = Command::new("boring-name");
    command_with_env_and_cwd.current_dir("/some/path").env("FOO", "bar");
    assert_eq!(
        format!("{command_with_env_and_cwd:?}"),
        r#"cd "/some/path" && FOO="bar" "boring-name""#
    );
    assert_eq!(
        format!("{command_with_env_and_cwd:#?}"),
        format!(
            r#"Command {{
    program: "boring-name",
    args: [
        "boring-name",
    ],
    env: CommandEnv {{
        clear: false,
        vars: {{
            "FOO": Some(
                "bar",
            ),
        }},
    }},
    cwd: Some(
        "/some/path",
    ),
{PIDFD}}}"#
        )
    );

    let mut command_with_removed_env = Command::new("boring-name");
    command_with_removed_env.env_remove("FOO").env_remove("BAR");
    assert_eq!(format!("{command_with_removed_env:?}"), r#"env -u BAR -u FOO "boring-name""#);
    assert_eq!(
        format!("{command_with_removed_env:#?}"),
        format!(
            r#"Command {{
    program: "boring-name",
    args: [
        "boring-name",
    ],
    env: CommandEnv {{
        clear: false,
        vars: {{
            "BAR": None,
            "FOO": None,
        }},
    }},
{PIDFD}}}"#
        )
    );

    let mut command_with_cleared_env = Command::new("boring-name");
    command_with_cleared_env.env_clear().env("BAR", "val").env_remove("FOO");
    assert_eq!(format!("{command_with_cleared_env:?}"), r#"env -i BAR="val" "boring-name""#);
    assert_eq!(
        format!("{command_with_cleared_env:#?}"),
        format!(
            r#"Command {{
    program: "boring-name",
    args: [
        "boring-name",
    ],
    env: CommandEnv {{
        clear: true,
        vars: {{
            "BAR": Some(
                "val",
            ),
        }},
    }},
{PIDFD}}}"#
        )
    );
}

// See issue #91991
#[test]
#[cfg(windows)]
fn run_bat_script() {
    let tempdir = crate::test_helpers::tmpdir();
    let script_path = tempdir.join("hello.cmd");

    crate::fs::write(&script_path, "@echo Hello, %~1!").unwrap();
    let output = Command::new(&script_path)
        .arg("fellow Rustaceans")
        .stdout(crate::process::Stdio::piped())
        .spawn()
        .unwrap()
        .wait_with_output()
        .unwrap();
    assert!(output.status.success());
    assert_eq!(String::from_utf8_lossy(&output.stdout).trim(), "Hello, fellow Rustaceans!");
}

// See issue #95178
#[test]
#[cfg(windows)]
fn run_canonical_bat_script() {
    let tempdir = crate::test_helpers::tmpdir();
    let script_path = tempdir.join("hello.cmd");

    crate::fs::write(&script_path, "@echo Hello, %~1!").unwrap();

    // Try using a canonical path
    let output = Command::new(&script_path.canonicalize().unwrap())
        .arg("fellow Rustaceans")
        .stdout(crate::process::Stdio::piped())
        .spawn()
        .unwrap()
        .wait_with_output()
        .unwrap();
    assert!(output.status.success());
    assert_eq!(String::from_utf8_lossy(&output.stdout).trim(), "Hello, fellow Rustaceans!");
}

#[test]
fn terminate_exited_process() {
    let mut cmd = if cfg!(target_os = "android") {
        let mut p = shell_cmd();
        p.args(&["-c", "true"]);
        p
    } else {
        known_command()
    };
    let mut p = cmd.stdout(Stdio::null()).spawn().unwrap();
    p.wait().unwrap();
    assert!(p.kill().is_ok());
    assert!(p.kill().is_ok());
}
