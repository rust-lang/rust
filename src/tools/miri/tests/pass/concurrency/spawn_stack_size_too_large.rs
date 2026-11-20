//@ignore-target: windows # FIXME: `CreateThread` shim ignores the stack size
//@run-native

use std::thread;

fn main() {
    // A stack this big can never be reserved, so spawning must fail instead of panicking.
    let result = thread::Builder::new().stack_size(usize::MAX).spawn(|| {});
    assert!(result.is_err());

    // A normal stack size still works.
    thread::Builder::new().stack_size(1 << 20).spawn(|| {}).unwrap().join().unwrap();
}
