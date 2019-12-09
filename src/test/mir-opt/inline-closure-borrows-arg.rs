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
// fn foo(_1: T, _2: &i32) -> i32{
//     debug _t => _1;
//     debug q => _2;
//     let mut _0: i32;
//     let _3: [closure@foo<T>::{{closure}}#0];
//     let mut _4: &[closure@foo<T>::{{closure}}#0];
//     let mut _5: (&i32, &i32);
//     let mut _6: &i32;
//     let mut _7: &i32;
//     let mut _8: &i32;
//     let mut _9: &i32;
//     scope 1 {
//         debug x => _3;
//         scope 2 {
//             debug r => _8;
//             debug _s => _9;
//         }
//     }
//     scope 3 {
//         debug variable => _8;
//     }
//     bb0: {
//         ...
//         _3 = [closure@foo::<T>::{{closure}}#0];
//         ...
//         _4 = &_3;
//         ...
//         _6 = &(*_2);
//         ...
//         _7 = &(*_2);
//         _5 = (move _6, move _7);
//         _8 = move (_5.0: &i32);
//         _9 = move (_5.1: &i32);
//         ...
//         _0 = (*_8);
//         ...
//         return;
//     }
// }
// END rustc.foo.Inline.after.mir
