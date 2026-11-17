//@only-target: windows # this directly tests windows-only functions
//@revisions: with_isolation without_isolation
//@[without_isolation] compile-flags: -Zmiri-disable-isolation
//@run-native

use windows_sys::Win32::System::Threading::GetCurrentProcessId;

fn main() {
    let pid = unsafe { GetCurrentProcessId() };
    assert_ne!(pid, 0);
    assert_eq!(pid, std::process::id());

    std::thread::spawn(move || {
        assert_eq!(unsafe { GetCurrentProcessId() }, pid);
    })
    .join()
    .unwrap();
}
