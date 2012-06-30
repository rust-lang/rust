// Test rules governing higher-order pure fns.

pure fn range(from: uint, to: uint, f: fn(uint)) {
    let mut i = from;
    while i < to {
        f(i); // Note: legal to call argument, even if it is not pure.
        i += 1u;
    }
}

pure fn range2(from: uint, to: uint, f: fn(uint)) {
    do range(from, to) |i| {
        f(i*2u);
    }
}

pure fn range3(from: uint, to: uint, f: fn(uint)) {
    range(from, to, f)
}

pure fn range4(from: uint, to: uint) {
    range(from, to, print) //! ERROR access to impure function prohibited in pure context
}

pure fn range5(from: uint, to: uint, x: {f: fn(uint)}) {
    range(from, to, x.f) //! ERROR access to impure function prohibited in pure context
}

pure fn range6(from: uint, to: uint, x: @{f: fn(uint)}) {
    range(from, to, x.f) //! ERROR access to impure function prohibited in pure context
}

pure fn range7(from: uint, to: uint) {
    do range(from, to) |i| {
        print(i); //! ERROR access to impure function prohibited in pure context
    }
}

pure fn range8(from: uint, to: uint) {
    range(from, to, noop);
}

fn print(i: uint) { #error["i=%u", i]; }

pure fn noop(_i: uint) {}

fn main() {
}