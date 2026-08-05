const fn nested(x: (for<'a> fn(&'a ()), String)) -> (fn(&'static ()), String) {
    x //~ ERROR mismatched types
}

pub const TEST: (fn(&'static ()), String) = nested((|_x| (), String::new()));

fn main() {}
