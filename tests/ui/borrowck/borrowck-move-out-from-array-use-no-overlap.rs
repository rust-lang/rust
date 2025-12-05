//@ check-pass

fn array() -> [(String, String); 3] {
    Default::default()
}

// Const Index + Const Index

fn move_out_from_begin_and_one_from_end() {
    let a = array();
    let [_, _, _x] = a;
    let [.., ref _y, _] = a;
}

fn move_out_from_begin_field_and_end_field() {
    let a = array();
    let [_, _, (_x, _)] = a;
    let [.., (_, ref _y)] = a;
}

// Const Index + Slice

fn move_out_by_const_index_and_subslice() {
    let a = array();
    let [_x, _, _] = a;
    let [_, ref _y @ ..] = a;
}

fn move_out_by_const_index_end_and_subslice() {
    let a = array();
    let [.., _x] = a;
    let [ref _y @ .., _] = a;
}

fn move_out_by_const_index_field_and_subslice() {
    let a = array();
    let [(_x, _), _, _] = a;
    let [_, ref _y @ ..] = a;
}

fn move_out_by_const_index_end_field_and_subslice() {
    let a = array();
    let [.., (_x, _)] = a;
    let [ref _y @ .., _] = a;
}

fn move_out_by_const_subslice_and_index_field() {
    let a = array();
    let [_, _y @ ..] = a;
    let [(ref _x, _), _, _] = a;
}

fn move_out_by_const_subslice_and_end_index_field() {
    let a = array();
    let [_y @ .., _] = a;
    let [.., (ref _x, _)] = a;
}

// Slice + Slice

fn move_out_by_subslice_and_subslice() {
    let a = array();
    let [x @ .., _, _] = a;
    let [_, ref _y @ ..] = a;
}

fn main() {}
