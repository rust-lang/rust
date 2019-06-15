// Basic test for liveness constraints: the region (`R1`) that appears
// in the type of `p` includes the points after `&v[0]` up to (but not
// including) the call to `use_x`. The `else` branch is not included.

// compile-flags:-Zborrowck=mir -Zverbose
//                              ^^^^^^^^^ force compiler to dump more region information

#![allow(warnings)]

fn use_x(_: usize) -> bool { true }

fn main() {
    let mut v = [1, 2, 3];
    let p = &v[0];
    let q = p;
    if true {
        use_x(*q);
    } else {
        use_x(22);
    }
}

// END RUST SOURCE
// START rustc.main.nll.0.mir
// | '_#2r | U0 | {bb2[0..=8], bb3[0], bb5[0..=2]}
// | '_#3r | U0 | {bb2[1..=8], bb3[0], bb5[0..=2]}
// | '_#4r | U0 | {bb2[4..=8], bb3[0], bb5[0..=2]}
// END rustc.main.nll.0.mir
// START rustc.main.nll.0.mir
// let _2: &'_#3r usize;
// ...
// let _6: &'_#4r usize;
// ...
// _2 = &'_#2r _1[_3];
// ...
// _6 = _2;
// END rustc.main.nll.0.mir
