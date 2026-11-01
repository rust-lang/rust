fn main() {}

struct X;

impl X {
    const Y: u8; //~ ERROR associated constant in `impl` without body
}
