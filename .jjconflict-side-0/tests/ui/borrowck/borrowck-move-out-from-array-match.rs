fn array() -> [(String, String); 3] {
    Default::default()
}

// Const Index + Const Index

fn move_out_from_begin_and_end() {
    let a = array();
    match a {
        [_, _, _x] => {}
    }
    match a {
        [.., _y] => {} //~ ERROR use of moved value
    }
}

fn move_out_from_begin_field_and_end() {
    let a = array();
    match a {
        [_, _, (_x, _)] => {}
    }
    match a {
        [.., _y] => {} //~ ERROR use of partially moved value
    }
}

fn move_out_from_begin_field_and_end_field() {
    let a = array();
    match a {
        [_, _, (_x, _)] => {}
    }
    match a {
        [.., (_y, _)] => {} //~ ERROR use of moved value
    }
}

// Const Index + Slice

fn move_out_by_const_index_and_subslice() {
    let a = array();
    match a {
        [_x, _, _] => {}
    }
    match a {
        [_y @ .., _, _] => {}
        //~^ ERROR use of partially moved value
    }
}

fn move_out_by_const_index_end_and_subslice() {
    let a = array();
    match a {
        [.., _x] => {}
    }
    match a {
        [_, _, _y @ ..] => {}
        //~^ ERROR use of partially moved value
    }
}

fn move_out_by_const_index_field_and_subslice() {
    let a = array();
    match a {
        [(_x, _), _, _] => {}
    }
    match a {
        [_y @ .., _, _] => {}
        //~^ ERROR use of partially moved value
    }
}

fn move_out_by_const_index_end_field_and_subslice() {
    let a = array();
    match a {
        [.., (_x, _)] => {}
    }
    match a {
        [_, _, _y @ ..] => {}
        //~^ ERROR use of partially moved value
    }
}

fn move_out_by_subslice_and_const_index_field() {
    let a = array();
    match a {
        [_y @ .., _, _] => {}
    }
    match a {
        [(_x, _), _, _] => {} //~ ERROR use of moved value
    }
}

fn move_out_by_subslice_and_const_index_end_field() {
    let a = array();
    match a {
        [_, _, _y @ ..] => {}
    }
    match a {
        [.., (_x, _)] => {} //~ ERROR use of moved value
    }
}

// Slice + Slice

fn move_out_by_subslice_and_subslice() {
    let a = array();
    match a {
        [x @ .., _] => {}
    }
    match a {
        [_, _y @ ..] => {}
        //~^ ERROR use of partially moved value
    }
}

fn main() {}
