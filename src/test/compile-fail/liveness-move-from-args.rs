fn take(-_x: int) { }

fn from_by_value_arg(++x: int) {
    take(x);  //~ ERROR illegal move from argument `x`, which is not copy or move mode
}

fn from_by_mut_ref_arg(&x: int) {
    take(x);  //~ ERROR illegal move from argument `x`, which is not copy or move mode
}

fn from_by_ref_arg(&&x: int) {
    take(x);  //~ ERROR illegal move from argument `x`, which is not copy or move mode
}

fn from_copy_arg(+x: int) {
    take(x);
}

fn from_move_arg(-x: int) {
    take(x);
}

fn main() {
}
