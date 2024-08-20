macro_rules! foo {
    ($p:expr) => {
        if let $p = Some(42) { //~ ERROR expected pattern, found expression `Some(3)`
            return;
        }
    };
}

fn main() {
    foo!(Some(3));
}
