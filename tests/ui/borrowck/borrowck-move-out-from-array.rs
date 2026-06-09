fn array() -> [(String, String); 3] {
    Default::default()
}

// Const Index + Const Index

fn move_out_from_begin_and_end() {
    let a = array();
    let [_, _, _x] = a;
    let [.., _y] = a; //~ ERROR [E0382]
}

fn move_out_from_begin_field_and_end() {
    let a = array();
    let [_, _, (_x, _)] = a;
    let [.., _y] = a; //~ ERROR [E0382]
}

fn move_out_from_begin_field_and_end_field() {
    let a = array();
    let [_, _, (_x, _)] = a;
    let [.., (_y, _)] = a; //~ ERROR [E0382]
}

// Const Index + Slice

fn move_out_by_const_index_and_subslice() {
    let a = array();
    let [_x, _, _] = a;
    let [_y @ .., _, _] = a; //~ ERROR [E0382]
}

fn move_out_by_const_index_end_and_subslice() {
    let a = array();
    let [.., _x] = a;
    let [_, _, _y @ ..] = a; //~ ERROR [E0382]
}

fn move_out_by_const_index_field_and_subslice() {
    let a = array();
    let [(_x, _), _, _] = a;
    let [_y @ .., _, _] = a; //~ ERROR [E0382]
}

fn move_out_by_const_index_end_field_and_subslice() {
    let a = array();
    let [.., (_x, _)] = a;
    let [_, _, _y @ ..] = a; //~ ERROR [E0382]
}

fn move_out_by_subslice_and_const_index_field() {
    let a = array();
    let [_y @ .., _, _] = a;
    let [(_x, _), _, _] = a; //~ ERROR [E0382]
}

fn move_out_by_subslice_and_const_index_end_field() {
    let a = array();
    let [_, _, _y @ ..] = a;
    let [.., (_x, _)] = a; //~ ERROR [E0382]
}

// Slice + Slice

fn move_out_by_subslice_and_subslice() {
    let a = array();
    let [x @ .., _] = a;
    let [_, _y @ ..] = a; //~ ERROR [E0382]
}

fn main() {}
