//@ compile-flags: -Zmir-opt-level=0 -Zmir-enable-passes=+ScalarReplacementOfAggregates -Aunused

struct Inner {
    value: u8,
}

struct Outer {
    inner: Inner,
}

fn main() {
    let x = Outer { inner: Inner { value: 7 } };
}
