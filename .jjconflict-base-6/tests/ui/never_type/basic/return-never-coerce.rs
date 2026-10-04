// Test that ! coerces to other types.

//@ run-fail
//@ error-pattern:aah!
//@ needs-subprocess

fn call_another_fn<T, F: FnOnce() -> T>(f: F) -> T {
    f()
}

fn wub() -> ! {
    panic!("aah!");
}

fn main() {
    let x: i32 = call_another_fn(wub);
    let y: u32 = wub();
}
