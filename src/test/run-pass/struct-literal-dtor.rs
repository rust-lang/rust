struct foo {
    x: ~str,
}

impl foo : Drop {
    fn finalize() {
        error!("%s", self.x);
    }
}

fn main() {
    let _z = foo { x: ~"Hello" };
}
