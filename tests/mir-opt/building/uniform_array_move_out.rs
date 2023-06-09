#![feature(stmt_expr_attributes, rustc_attrs)]

// EMIT_MIR uniform_array_move_out.move_out_from_end.built.after.mir
fn move_out_from_end() {
    let a = [
        #[rustc_box]
        Box::new(1),
        #[rustc_box]
        Box::new(2),
    ];
    let [.., _y] = a;
}

// EMIT_MIR uniform_array_move_out.move_out_by_subslice.built.after.mir
fn move_out_by_subslice() {
    let a = [
        #[rustc_box]
        Box::new(1),
        #[rustc_box]
        Box::new(2),
    ];
    let [_y @ ..] = a;
}

fn main() {
    move_out_by_subslice();
    move_out_from_end();
}
