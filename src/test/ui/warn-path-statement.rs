// compile-flags: -D path-statements
struct Droppy;

impl Drop for Droppy {
    fn drop(&mut self) {}
}

fn main() {
    let x = 10;
    x; //~ ERROR path statement with no effect

    let y = Droppy;
    y; //~ ERROR path statement drops value

    let z = (Droppy,);
    z; //~ ERROR path statement drops value
}
