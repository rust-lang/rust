static A1: usize = 1;
static mut A2: usize = 1;
const A3: usize = 1;

fn main() {
    match 1 {
        A1 => {} //~ ERROR: match bindings cannot shadow statics
        A2 => {} //~ ERROR: match bindings cannot shadow statics
        A3 => {}
        _ => {}
    }
}
