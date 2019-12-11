#![feature(box_syntax)]
#![feature(slice_patterns)]

fn move_out_from_end() {
    let a = [box 1, box 2];
    let [.., _y] = a;
}

fn move_out_by_subslice() {
    let a = [box 1, box 2];
    let [_y @ ..] = a;
}

fn main() {
    move_out_by_subslice();
    move_out_from_end();
}

// END RUST SOURCE

// START rustc.move_out_from_end.mir_map.0.mir
//      _6 = move _1[1 of 2];
//      _0 = ();
// END rustc.move_out_from_end.mir_map.0.mir

// START rustc.move_out_by_subslice.mir_map.0.mir
//     _6 = move _1[0..2];
//     _0 = ();
// END rustc.move_out_by_subslice.mir_map.0.mir
