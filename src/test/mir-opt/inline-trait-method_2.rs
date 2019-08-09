// compile-flags: -Z span_free_formats -Z mir-opt-level=3

fn test2(x: &dyn X) -> bool {
    test(x)
}

#[inline]
fn test(x: &dyn X) -> bool {
    x.y()
}

trait X {
    fn y(&self) -> bool {
        false
    }
}

impl X for () {
    fn y(&self) -> bool {
        true
    }
}

fn main() {
    println!("Should be true: {}", test2(&()));
}

// END RUST SOURCE
// START rustc.test2.Inline.after.mir
// ...
// bb0: {
// ...
//     _0 = const <dyn X as X>::y(move _2) -> bb1;
// }
// ...
// END rustc.test2.Inline.after.mir
