fn main() {}

struct X;

impl X {
    fn f(); //~ ERROR associated function in `impl` without body
}
