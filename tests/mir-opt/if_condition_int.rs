// skip-filecheck
//@ test-mir-pass: SimplifyComparisonIntegral
// EMIT_MIR if_condition_int.opt_u32.SimplifyComparisonIntegral.diff
// EMIT_MIR if_condition_int.opt_negative.SimplifyComparisonIntegral.diff
// EMIT_MIR if_condition_int.opt_char.SimplifyComparisonIntegral.diff
// EMIT_MIR if_condition_int.opt_i8.SimplifyComparisonIntegral.diff
// EMIT_MIR if_condition_int.dont_opt_bool.SimplifyComparisonIntegral.diff
// EMIT_MIR if_condition_int.opt_multiple_ifs.SimplifyComparisonIntegral.diff
// EMIT_MIR if_condition_int.dont_remove_comparison.SimplifyComparisonIntegral.diff
// EMIT_MIR if_condition_int.dont_opt_floats.SimplifyComparisonIntegral.diff

fn opt_u32(x: u32) -> u32 {
    if x == 42 { 0 } else { 1 }
}

// don't opt: it is already optimal to switch on the bool
fn dont_opt_bool(x: bool) -> u32 {
    if x { 0 } else { 1 }
}

fn opt_char(x: char) -> u32 {
    if x == 'x' { 0 } else { 1 }
}

fn opt_i8(x: i8) -> u32 {
    if x == 42 { 0 } else { 1 }
}

fn opt_negative(x: i32) -> u32 {
    if x == -42 { 0 } else { 1 }
}

fn opt_multiple_ifs(x: u32) -> u32 {
    if x == 42 {
        0
    } else if x != 21 {
        1
    } else {
        2
    }
}

// test that we optimize, but do not remove the b statement, as that is used later on
fn dont_remove_comparison(a: i8) -> i32 {
    let b = a == 17;
    match b {
        false => 10 + b as i32,
        true => 100 + b as i32,
    }
}

// test that we do not optimize on floats
fn dont_opt_floats(a: f32) -> i32 {
    if a == -42.0 { 0 } else { 1 }
}

fn main() {
    opt_u32(0);
    opt_char('0');
    opt_i8(22);
    dont_opt_bool(false);
    opt_negative(0);
    opt_multiple_ifs(0);
    dont_remove_comparison(11);
    dont_opt_floats(1.0);
}
