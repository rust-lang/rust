#![feature(box_syntax)]
#![feature(slice_patterns)]

fn move_out_from_begin_and_end() {
    let a = [box 1, box 2];
    let [_, _x] = a;
    let [.., _y] = a; //~ ERROR [E0382]
}

fn move_out_by_const_index_and_subslice() {
    let a = [box 1, box 2];
    let [_x, _] = a;
    let [_y..] = a; //~ ERROR [E0382]
}

fn main() {}
