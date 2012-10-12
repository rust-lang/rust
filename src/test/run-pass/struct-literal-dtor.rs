struct foo {
    x: ~str,
    drop { error!("%s", self.x); }
}

fn main() {
    let _z = foo { x: ~"Hello" };
}
