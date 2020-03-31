// FIXME: missing sysroot spans (#53081)
// ignore-i586-unknown-linux-gnu
// ignore-i586-unknown-linux-musl
// ignore-i686-unknown-linux-musl

struct C {
    x: isize,
}

impl Drop for C {
    fn drop(&mut self) {
        println!("dropping: {}", self.x);
    }
}

fn main() {
    let c = C{ x: 2};
    let _d = c.clone(); //~ ERROR no method named `clone` found
}
