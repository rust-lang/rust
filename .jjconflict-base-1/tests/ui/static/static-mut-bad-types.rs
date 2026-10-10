static mut a: isize = 3;

fn main() {
    unsafe {
        a = true; //~ ERROR: mismatched types
    }
}
