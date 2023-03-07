//@only-target-linux
//@only-on-host
//@normalize-stderr-test: "OS `.*`" -> "$$OS"

extern "C" {
    fn foo();
}

fn main() {
    unsafe {
        foo(); //~ ERROR: unsupported operation: can't call foreign function `foo`
    }
}
