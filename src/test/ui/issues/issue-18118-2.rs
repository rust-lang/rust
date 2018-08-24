pub fn main() {
    const z: &'static isize = {
        static p: isize = 3;
        &p
        //~^ ERROR constants cannot refer to statics, use a constant instead
    };
}
