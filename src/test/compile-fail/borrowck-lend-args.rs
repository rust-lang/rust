fn borrow(_v: &int) {}

fn borrow_from_arg_imm_ref(&&v: ~int) {
    borrow(v);
}

fn borrow_from_arg_mut_ref(v: &mut ~int) {
    borrow(*v); //~ ERROR illegal borrow unless pure
    //~^ NOTE impure due to access to impure function
}

fn borrow_from_arg_move(-v: ~int) {
    borrow(v);
}

fn borrow_from_arg_copy(+v: ~int) {
    borrow(v);
}

fn borrow_from_arg_val(++v: ~int) {
    borrow(v);
}

fn main() {
}
