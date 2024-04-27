pub fn main() {
    const z: &'static isize = {
        static p: isize = 3;
        &p //~ ERROR referencing statics
    };
}
