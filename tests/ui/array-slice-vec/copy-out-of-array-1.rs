//@ run-pass

// Ensure that we can copy out of a fixed-size array.
//
// (Compare with ui/moves/move-out-of-array-1.rs)

#[derive(Copy, Clone)]
struct C { _x: u8 }

fn main() {
    fn d() -> C { C { _x: 0 } }

    let _d1 = foo([d(), d(), d(), d()], 1);
    let _d3 = foo([d(), d(), d(), d()], 3);
}

fn foo(a: [C; 4], i: usize) -> C {
    a[i]
}
