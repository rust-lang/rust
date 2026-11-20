//@ check-pass
// Due to #53114, which causes a "read" of the `_` patterns,
// the borrow-checker refuses this code, while it should probably be allowed.
// Once the bug is fixed, the test, which is derived from a
// passing test for `let` statements, should become check-pass.

fn array() -> [(String, String); 3] {
    Default::default()
}

// Const Index + Const Index

fn move_out_from_begin_and_one_from_end() {
    let a = array();
    match a {
        [_, _, _x] => {}
    }
    match a {
        [.., ref _y, _] => {}
    }
}

fn move_out_from_begin_field_and_end_field() {
    let a = array();
    match a {
        [_, _, (_x, _)] => {}
    }
    match a {
        [.., (_, ref _y)] => {}
    }
}

// Const Index + Slice

fn move_out_by_const_index_and_subslice() {
    let a = array();
    match a {
        [_x, _, _] => {}
    }
    match a {
        [_, ref _y @ ..] => {}
    }
}

fn move_out_by_const_index_end_and_subslice() {
    let a = array();
    match a {
        [.., _x] => {}
    }
    match a {
        [ref _y @ .., _] => {}
    }
}

fn move_out_by_const_index_field_and_subslice() {
    let a = array();
    match a {
        [(_x, _), _, _] => {}
    }
    match a {
        [_, ref _y @ ..] => {}
    }
}

fn move_out_by_const_index_end_field_and_subslice() {
    let a = array();
    match a {
        [.., (_x, _)] => {}
    }
    match a {
        [ref _y @ .., _] => {}
    }
}

fn move_out_by_const_subslice_and_index_field() {
    let a = array();
    match a {
        [_, _y @ ..] => {}
    }
    match a {
        [(ref _x, _), _, _] => {}
    }
}

fn move_out_by_const_subslice_and_end_index_field() {
    let a = array();
    match a {
        [_y @ .., _] => {}
    }
    match a {
        [.., (ref _x, _)] => {}
    }
}

// Slice + Slice

fn move_out_by_subslice_and_subslice() {
    let a = array();
    match a {
        [x @ .., _, _] => {}
    }
    match a {
        [_, ref _y @ ..] => {}
    }
}

fn main() {}
