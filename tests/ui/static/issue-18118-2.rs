pub fn main() {
    const Z: &'static isize = {
        static P: isize = 3;
        &P //~ ERROR constants cannot refer to statics
    };
}
