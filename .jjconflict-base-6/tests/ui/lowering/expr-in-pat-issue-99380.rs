macro_rules! foo {
    ($p:expr) => {
        if let $p = Some(42) {
            return;
        }
    };
}

macro_rules! custom_matches {
    ($e:expr, $p:expr) => {
        match $e {
            $p => true,
            _ => false,
        }
    };
}

fn main() {
    foo!(Some(3)); //~ ERROR arbitrary expressions aren't allowed in patterns

    let _ = custom_matches!(67, 6 | 7); //~ ERROR arbitrary expressions aren't allowed in patterns
}
