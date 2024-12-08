//@revisions: with_isolation without_isolation
//@[without_isolation] compile-flags: -Zmiri-disable-isolation

fn getpid() -> u32 {
    std::process::id()
}

fn main() {
    let pid = getpid();

    std::thread::spawn(move || {
        assert_eq!(getpid(), pid);
    });

    // Test that in isolation mode a deterministic value will be returned.
    // The value 1000 is not important, we only care that whatever the value
    // is, won't change from execution to execution.
    #[cfg(with_isolation)]
    assert_eq!(pid, 1000);
}
