struct foo {
    x: ~str,
}

impl foo : Drop {
    fn finalize(&self) {
        error!("%s", self.x);
    }
}

fn main() {
    let _z = foo { x: ~"Hello" };
}
