struct thing<'a, Q:'a> {
    x: &'a Q
}

fn thing<'a,Q>(x: &Q) -> thing<'a,Q> {
    thing{ x: x } //~ ERROR 16:5: 16:18: explicit lifetime required in the type of `x` [E0621]
}

fn main() {
    thing(&());
}
