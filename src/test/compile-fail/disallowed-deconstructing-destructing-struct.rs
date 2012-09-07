struct X {
    x: ~str,
    drop {
        error!("value: %s", self.x);
    }
}

fn unwrap(+x: X) -> ~str {
    let X { x: y } = x; //~ ERROR deconstructing struct not allowed in pattern
    y
}

fn main() {
    let x = X { x: ~"hello" };
    let y = unwrap(x);
    error!("contents: %s", y);
}
