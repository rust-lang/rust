//@ run-pass
#![allow(dead_code)]


fn borrow(_v: &isize) {}

fn borrow_from_arg_imm_ref(v: Box<isize>) {
    borrow(&*v);
}

fn borrow_from_arg_mut_ref(v: &mut Box<isize>) {
    borrow(&**v);
}

fn borrow_from_arg_copy(v: Box<isize>) {
    borrow(&*v);
}

pub fn main() {
}
