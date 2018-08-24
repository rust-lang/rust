#![feature(rustc_attrs)]

#[rustc_error]
fn main() { //~ ERROR compilation successful
    if ('x' as char) < ('y' as char) {
        print!("x");
    } else {
        print!("y");
    }
}
