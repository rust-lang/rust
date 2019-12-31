// compile-flags: -Z span_free_formats

// Tests that MIR inliner can handle closure captures.

fn main() {
    println!("{:?}", foo(0, 14));
}

fn foo<T: Copy>(t: T, q: i32) -> (i32, T) {
    let x = |_q| (q, t);
    x(q)
}

// END RUST SOURCE
// START rustc.foo.Inline.after.mir
// fn foo(_1: T, _2: i32) -> (i32, T){
//     debug t => _1;
//     debug q => _2;
//     let mut _0: (i32, T);
//     let _3: [closure@foo<T>::{{closure}}#0 q:&i32, t:&T];
//     let mut _4: &i32;
//     let mut _5: &T;
//     let mut _6: &[closure@foo<T>::{{closure}}#0 q:&i32, t:&T];
//     let mut _7: (i32,);
//     let mut _8: i32;
//     let mut _11: i32;
//     scope 1 {
//         debug x => _3;
//         scope 2 {
//             debug _q => _11;
//             debug q => (*((*_6).0: &i32));
//             debug t => (*((*_6).1: &T));
//             let mut _9: i32;
//             let mut _10: T;
//         }
//     }
//     bb0: {
//         ...
//         _4 = &_2;
//         ...
//         _5 = &_1;
//         _3 = [closure@foo::<T>::{{closure}}#0] { q: move _4, t: move _5 };
//         ...
//         _6 = &_3;
//         ...
//         ...
//         _8 = _2;
//         _7 = (move _8,);
//         _11 = move (_7.0: i32);
//         ...
//         _9 = (*((*_6).0: &i32));
//         ...
//         _10 = (*((*_6).1: &T));
//         (_0.0: i32) = move _9;
//         (_0.1: T) = move _10;
//         ...
//         return;
//     }
// }
// END rustc.foo.Inline.after.mir
