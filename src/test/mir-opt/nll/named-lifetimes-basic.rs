// Basic test for named lifetime translation. Check that we
// instantiate the types that appear in function arguments with
// suitable variables and that we setup the outlives relationship
// between R0 and R1 properly.

// compile-flags:-Zborrowck=mir -Zverbose
//                              ^^^^^^^^^ force compiler to dump more region information

#![allow(warnings)]

fn use_x<'a, 'b: 'a, 'c>(w: &'a mut i32, x: &'b u32, y: &'a u32, z: &'c u32) -> bool { true }

fn main() {
}

// END RUST SOURCE
// START rustc.use_x.nll.0.mir
// | Free Region Mapping
// | '_#0r    | Global   | ['_#2r, '_#1r, '_#0r, '_#4r, '_#3r]
// | '_#1r    | External | ['_#1r, '_#4r]
// | '_#2r    | External | ['_#2r, '_#1r, '_#4r]
// | '_#3r    | Local    | ['_#4r, '_#3r]
// | '_#4r    | Local    | ['_#4r]
// |
// | Inferred Region Values
// | '_#0r    | U0 | {bb0[0..=1], '_#0r, '_#1r, '_#2r, '_#3r, '_#4r}
// | '_#1r    | U0 | {bb0[0..=1], '_#1r}
// | '_#2r    | U0 | {bb0[0..=1], '_#2r}
// | '_#3r    | U0 | {bb0[0..=1], '_#3r}
// | '_#4r    | U0 | {bb0[0..=1], '_#4r}
// | '_#5r    | U0 | {bb0[0..=1], '_#1r}
// | '_#6r    | U0 | {bb0[0..=1], '_#2r}
// | '_#7r    | U0 | {bb0[0..=1], '_#1r}
// | '_#8r    | U0 | {bb0[0..=1], '_#3r}
// |
// ...
// fn use_x(_1: &'_#5r mut i32, _2: &'_#6r u32, _3: &'_#7r u32, _4: &'_#8r u32) -> bool {
// END rustc.use_x.nll.0.mir
