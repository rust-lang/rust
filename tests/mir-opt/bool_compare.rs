// unit-test: InstSimplify

// EMIT_MIR bool_compare.opt1.InstSimplify.diff
fn opt1(x: bool) -> u32 {
    if x != true { 0 } else { 1 }
}

// EMIT_MIR bool_compare.opt2.InstSimplify.diff
fn opt2(x: bool) -> u32 {
    if true != x { 0 } else { 1 }
}

// EMIT_MIR bool_compare.opt3.InstSimplify.diff
fn opt3(x: bool) -> u32 {
    if x == false { 0 } else { 1 }
}

// EMIT_MIR bool_compare.opt4.InstSimplify.diff
fn opt4(x: bool) -> u32 {
    if false == x { 0 } else { 1 }
}

fn main() {
    opt1(false);
    opt2(false);
    opt3(false);
    opt4(false);
}
