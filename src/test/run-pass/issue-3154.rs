struct thing<Q> {
    x: &Q
}

fn thing<Q>(x: &Q) -> thing<Q> {
    thing{ x: x }
}

fn main() {
    thing(&());
}