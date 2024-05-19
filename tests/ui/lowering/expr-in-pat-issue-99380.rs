macro_rules! foo {
    ($p:expr) => {
        if let $p = Some(42) {
            return;
        }
    };
}

fn main() {
    foo!(Some(3)); //~ ERROR arbitrary expressions aren't allowed in patterns
}
