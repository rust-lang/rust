// Only works on Unix targets
//@ignore-target: windows wasm
//@only-on-host
//@normalize-stderr-test: "OS `.*`" -> "$$OS"

extern "C" {
    fn not_exported();
}

fn main() {
    unsafe {
        not_exported(); //~ ERROR: unsupported operation: can't call foreign function `not_exported`
    }
}
