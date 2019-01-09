// compile-flags: -Z span_free_formats -Z mir-opt-level=3

#[inline]
fn test(x: &dyn X) -> bool {
    x.y()
}

fn test2(x: &dyn X) -> bool {
    test(x)
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
//     _0 = const X::y(move _2) -> bb1;
// }
// ...
// END rustc.test2.Inline.after.mir
