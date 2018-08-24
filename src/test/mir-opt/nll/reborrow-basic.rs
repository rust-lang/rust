// Basic test for reborrow constraints: the region (`R5`) that appears
// in the type of `r_a` must outlive the region (`R7`) that appears in
// the type of `r_b`

// compile-flags:-Zborrowck=mir -Zverbose
//                              ^^^^^^^^^ force compiler to dump more region information

#![allow(warnings)]

fn use_x(_: &mut i32) -> bool { true }

fn main() {
    let mut foo: i32     = 22;
    let r_a: &mut i32 = &mut foo;
    let r_b: &mut i32 = &mut *r_a;
    use_x(r_b);
}

// END RUST SOURCE
// START rustc.main.nll.0.mir
// | '_#7r    | U0 | {bb0[4], bb0[8..=17]}
// ...
// | '_#9r    | U0 | {bb0[10], bb0[14..=17]}
// ...
// let _4: &'_#9r mut i32;
// ...
// let _2: &'_#7r mut i32;
// END rustc.main.nll.0.mir
