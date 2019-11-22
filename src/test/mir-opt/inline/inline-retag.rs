// compile-flags: -Z span_free_formats -Z mir-emit-retag

// Tests that MIR inliner fixes up `Retag`'s `fn_entry` flag

fn main() {
    println!("{}", bar());
}

fn bar() -> bool {
    let f = foo;
    f(&1, &-1)
}

#[inline(always)]
fn foo(x: &i32, y: &i32) -> bool {
    *x == *y
}

// END RUST SOURCE
// START rustc.bar.Inline.after.mir
// ...
//     bb0: {
//         ...
//         Retag(_3);
//         ...
//         Retag(_3);
//         Retag(_6);
//         StorageLive(_11);
//         _11 = (*_3);
//         StorageLive(_12);
//         _12 = (*_6);
//         _0 = Eq(move _11, move _12);
//         ...
//         return;
//     }
// ...
// END rustc.bar.Inline.after.mir
