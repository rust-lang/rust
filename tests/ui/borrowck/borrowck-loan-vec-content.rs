// Here we check that it is allowed to lend out an element of a
// (locally rooted) mutable, unique vector, and that we then prevent
// modifications to the contents.

fn takes_imm_elt<F>(_v: &isize, f: F) where F: FnOnce() {
    f();
}

fn has_mut_vec_and_does_not_try_to_change_it() {
    let mut v: Vec<isize> = vec![1, 2, 3];
    takes_imm_elt(&v[0], || {})
}

fn has_mut_vec_but_tries_to_change_it() {
    let mut v: Vec<isize> = vec![1, 2, 3];
    takes_imm_elt(
        &v[0],
        || { //~ ERROR cannot borrow `v` as mutable
            v[1] = 4;
        })
}

fn main() {
}
