// compile-flags: -Z span_free_formats

fn main() {
    println!("{}", test(&()));
}

fn test(x: &dyn X) -> u32 {
    x.y()
}

trait X {
    fn y(&self) -> u32 {
        1
    }
}

impl X for () {
    fn y(&self) -> u32 {
        2
    }
}

// END RUST SOURCE
// START rustc.test.Inline.after.mir
// ...
// bb0: {
// ...
//     _0 = const <dyn X as X>::y(move _2) -> bb1;
// }
// ...
// END rustc.test.Inline.after.mir
