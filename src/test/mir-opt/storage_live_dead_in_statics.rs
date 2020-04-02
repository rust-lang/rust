// Check that when we compile the static `XXX` into MIR, we do not
// generate `StorageStart` or `StorageEnd` statements.

// EMIT_MIR rustc.XXX.mir_map.0.mir
static XXX: &'static Foo = &Foo {
    tup: "hi",
    data: &[
        (0, 1), (0, 2), (0, 3),
        (0, 1), (0, 2), (0, 3),
        (0, 1), (0, 2), (0, 3),
        (0, 1), (0, 2), (0, 3),
        (0, 1), (0, 2), (0, 3),
        (0, 1), (0, 2), (0, 3),
        (0, 1), (0, 2), (0, 3),
        (0, 1), (0, 2), (0, 3),
        (0, 1), (0, 2), (0, 3),
        (0, 1), (0, 2), (0, 3),
        (0, 1), (0, 2), (0, 3),
        (0, 1), (0, 2), (0, 3),
        (0, 1), (0, 2), (0, 3),
        (0, 1), (0, 2), (0, 3),
    ]
};

#[derive(Debug)]
struct Foo {
    tup: &'static str,
    data: &'static [(u32, u32)]
}

fn main() {
    println!("{:?}", XXX);
}
