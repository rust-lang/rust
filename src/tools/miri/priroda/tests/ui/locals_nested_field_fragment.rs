//@ compile-flags: -Zmir-opt-level=0 -Zmir-enable-passes=+ScalarReplacementOfAggregates -Aunused

struct Leaf {
    field: u8,
}

struct Outer {
    inner: (Leaf,),
}

fn main() {
    let x = Outer { inner: (Leaf { field: 3 },) };
}
