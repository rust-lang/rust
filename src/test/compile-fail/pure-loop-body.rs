pure fn range(from: uint, to: uint, f: fn(uint) -> bool) {
    let mut i = from;
    while i < to {
        if !f(i) {ret;} // Note: legal to call argument, even if it is not pure.
        i += 1u;
    }
}

pure fn range2(from: uint, to: uint, f: fn(uint)) {
    for range(from, to) |i| {
        f(i*2u);
    }
}

pure fn range3(from: uint, to: uint, f: {x: fn(uint)}) {
    for range(from, to) |i| {
        f.x(i*2u); //! ERROR access to impure function prohibited
    }
}

fn main() {}
