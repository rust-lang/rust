// check-pass

macro_rules! make_item {
    () => { fn f() {} }
}

macro_rules! make_stmt {
    () => { let x = 0; }
}

fn f() {
    make_item! {}
}

fn g() {
    make_stmt! {}
}

fn main() {}
