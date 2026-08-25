fn takes_imm(x: &isize) { }

fn takes_mut(x: &mut isize) { }

fn apply<T, F>(t: T, f: F) where F: FnOnce(T) {
    f(t)
}

fn main() {
    apply(&3, takes_imm);
    apply(&3, takes_mut);
    //~^ ERROR type mismatch

    apply(&mut 3, takes_mut);
    apply(&mut 3, takes_imm);
    //~^ ERROR type mismatch
}
