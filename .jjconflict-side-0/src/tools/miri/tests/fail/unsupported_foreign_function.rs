//@normalize-stderr-test: "OS `.*`" -> "$$OS"

fn main() {
    extern "Rust" {
        fn foo();
    }

    unsafe {
        foo(); //~ ERROR: unsupported operation: can't call foreign function `foo`
    }
}
