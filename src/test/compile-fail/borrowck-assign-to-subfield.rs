fn main() {
    let mut p = {a: 1,
                 w: {a: 1},
                 x: @{a: 1},
                 y: @const {a: 1},
                 z: @mut{a: 1}};

    // even though `x` is not declared as a mutable field,
    // `p` as a whole is mutable, so it can be modified.
    p.a = 2;

    // this is true for an interior field too
    p.w.a = 2;

    // in these cases we pass through a box, so the mut
    // of the box is dominant
    p.x.a = 2;     //~ ERROR assigning to immutable field
    p.y.a = 2;     //~ ERROR assigning to const field
    p.z.a = 2;
}