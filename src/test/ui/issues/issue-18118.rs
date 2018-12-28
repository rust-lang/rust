pub fn main() {
    const z: &'static isize = {
        let p = 3;
        &p //~ ERROR `p` does not live long enough
    };
}
