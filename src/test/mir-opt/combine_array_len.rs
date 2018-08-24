fn norm2(x: [f32; 2]) -> f32 {
    let a = x[0];
    let b = x[1];
    a*a + b*b
}

fn main() {
    assert_eq!(norm2([3.0, 4.0]), 5.0*5.0);
}

// END RUST SOURCE

// START rustc.norm2.InstCombine.before.mir
//     _4 = Len(_1);
//     ...
//     _8 = Len(_1);
// END rustc.norm2.InstCombine.before.mir

// START rustc.norm2.InstCombine.after.mir
//     _4 = const 2usize;
//     ...
//     _8 = const 2usize;
// END rustc.norm2.InstCombine.after.mir
