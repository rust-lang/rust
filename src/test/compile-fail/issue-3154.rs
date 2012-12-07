struct thing<Q> {
    x: &Q
}

fn thing<Q>(x: &Q) -> thing<Q> {
    thing{ x: x } //~ ERROR cannot infer an appropriate lifetime
}

fn main() {
    thing(&());
}