//@ check-pass

pub fn main() {
    const Z: &'static isize = {
        static P: isize = 3;
        &P
    };
}
