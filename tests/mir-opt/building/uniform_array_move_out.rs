//@ compile-flags: -Zmir-opt-level=0
//@ needs-unwind
// skip-filecheck
#![feature(liballoc_internals, rustc_attrs)]

// EMIT_MIR uniform_array_move_out.move_out_from_end.built.after.mir
// EMIT_MIR uniform_array_move_out.move_out_from_end.ElaborateDrops.diff
fn move_out_from_end() {
    let a = [std::boxed::box_new(1), std::boxed::box_new(2)];
    let [.., _y] = a;
}

// EMIT_MIR uniform_array_move_out.move_out_from_middle.built.after.mir
// EMIT_MIR uniform_array_move_out.move_out_from_middle.ElaborateDrops.diff
fn move_out_from_middle() {
    let a = [std::boxed::box_new(1), std::boxed::box_new(2), std::boxed::box_new(3)];
    let [_, _y, _] = a;
}

// EMIT_MIR uniform_array_move_out.move_out_by_subslice.built.after.mir
// EMIT_MIR uniform_array_move_out.move_out_by_subslice.ElaborateDrops.diff
fn move_out_by_subslice() {
    let a = [std::boxed::box_new(1), std::boxed::box_new(2), std::boxed::box_new(3)];
    let [_y @ .., _] = a;
}

fn main() {
    move_out_by_subslice();
    move_out_from_end();
}
