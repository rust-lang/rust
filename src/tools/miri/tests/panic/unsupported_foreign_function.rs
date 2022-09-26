//@compile-flags: -Zmiri-panic-on-unsupported

fn main() {
    extern "Rust" {
        fn foo();
    }

    unsafe {
        foo();
    }
}
