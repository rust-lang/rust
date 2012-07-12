// xfail-test

const c_x: &blk/int = 22; //~ ERROR only the static region is allowed here
const c_y: &static/int = &22; //~ ERROR only the static region is allowed here

fn main() {
}