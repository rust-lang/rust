// compile-flags: -Z print-type-sizes
// build-pass
// ignore-pass

// At one point, zero-sized fields such as those in this file were causing
// incorrect output from `-Z print-type-sizes`.

#![feature(start)]

struct S1 {
    x: u32,
    y: u32,
    tag: (),
}

struct Void();
struct Empty {}

struct S5<TagW, TagZ> {
    tagw: TagW,
    w: u32,
    unit: (),
    x: u32,
    void: Void,
    y: u32,
    empty: Empty,
    z: u32,
    tagz: TagZ,
}

#[start]
fn start(_: isize, _: *const *const u8) -> isize {
    let _s1: S1 = S1 { x: 0, y: 0, tag: () };

    let _s5: S5<(), Empty> = S5 {
        tagw: (),
        w: 1,
        unit: (),
        x: 2,
        void: Void(),
        y: 3,
        empty: Empty {},
        z: 4,
        tagz: Empty {},
    };
    0
}
