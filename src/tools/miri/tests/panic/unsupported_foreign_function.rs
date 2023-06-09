//@compile-flags: -Zmiri-panic-on-unsupported
//@normalize-stderr-test: "OS `.*`" -> "$$OS"

fn main() {
    extern "Rust" {
        fn foo();
    }

    unsafe {
        foo();
    }
}
