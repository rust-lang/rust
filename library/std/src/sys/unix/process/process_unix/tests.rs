#[test]
fn test_command_fork_no_unwind() {
    use crate::os::unix::process::CommandExt;
    use crate::os::unix::process::ExitStatusExt;
    use crate::panic::catch_unwind;
    use crate::process::Command;

    let got = catch_unwind(|| {
        let mut c = Command::new("echo");
        c.arg("hi");
        unsafe {
            c.pre_exec(|| panic!("crash now!"));
        }
        let st = c.status().expect("failed to get command status");
        eprintln!("{:?}", st);
        st
    });
    eprintln!("got={:?}", &got);
    let estatus = got.expect("panic unexpectedly propagated");
    let signal = estatus.signal().expect("expected child to die of signal");
    assert!(signal == libc::SIGABRT || signal == libc::SIGILL);
}
