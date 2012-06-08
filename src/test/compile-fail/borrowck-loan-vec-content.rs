// Here we check that it is allowed to lend out an element of a
// (locally rooted) mutable, unique vector, and that we then prevent
// modifications to the contents.

fn takes_imm_elt(_v: &int, f: fn()) {
    f();
}

fn has_mut_vec_and_does_not_try_to_change_it() {
    let v = [mut 1, 2, 3];
    takes_imm_elt(&v[0]) {||
    }
}

fn has_mut_vec_but_tries_to_change_it() {
    let v = [mut 1, 2, 3];
    takes_imm_elt(&v[0]) {|| //! NOTE loan of mutable vec content granted here
        v[1] = 4; //! ERROR assigning to mutable vec content prohibited due to outstanding loan
    }
}

fn takes_const_elt(_v: &const int, f: fn()) {
    f();
}

fn has_mut_vec_and_tries_to_change_it() {
    let v = [mut 1, 2, 3];
    takes_const_elt(&const v[0]) {||
        v[1] = 4;
    }
}

fn main() {
}