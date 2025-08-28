//@ compile-flags: -Zmir-opt-level=0
// skip-filecheck
// Check that when we compile the static `XXX` into MIR, we do not
// generate `StorageStart` or `StorageEnd` statements.

// EMIT_MIR storage_live_dead_in_statics.XXX.built.after.mir
#[rustfmt::skip]
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
    ],
};

#[derive(Debug)]
struct Foo {
    tup: &'static str,
    data: &'static [(u32, u32)],
}

fn main() {
    println!("{:?}", XXX);
}
