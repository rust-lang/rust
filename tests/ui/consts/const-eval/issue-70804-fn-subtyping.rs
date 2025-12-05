//@ check-pass

const fn nested(x: (for<'a> fn(&'a ()), String)) -> (fn(&'static ()), String) {
    x
}

pub const TEST: (fn(&'static ()), String) = nested((|_x| (), String::new()));

fn main() {}
