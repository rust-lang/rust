// compile-flags: -Z span_free_formats

// Tests that MIR inliner can handle closure arguments,
// even when (#45894)

fn main() {
    println!("{}", foo(0, &14));
}

fn foo<T: Copy>(_t: T, q: &i32) -> i32 {
    let x = |r: &i32, _s: &i32| {
        let variable = &*r;
        *variable
    };
    x(q, q)
}

// END RUST SOURCE
// START rustc.foo.Inline.after.mir
// ...
// bb0: {
//     ...
//     _3 = [closure@HirId { owner: DefIndex(13), local_id: 31 }];
//     ...
//     _4 = &_3;
//     ...
//     _6 = &(*_2);
//     ...
//     _7 = &(*_2);
//     _5 = (move _6, move _7);
//     _8 = move (_5.0: &i32);
//     _9 = move (_5.1: &i32);
//     ...
//     _0 = (*_8);
//     ...
//     return;
// }
// ...
// END rustc.foo.Inline.after.mir
