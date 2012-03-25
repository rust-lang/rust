fn takes_mut(&&x: @mut int) { }
fn takes_const(&&x: @const int) { }
fn takes_imm(&&x: @int) { }

fn apply<T>(t: T, f: fn(T)) {
    f(t)
}

fn main() {
    apply(@3, takes_mut); //! ERROR (values differ in mutability)
    apply(@3, takes_const);
    apply(@3, takes_imm);

    apply(@mut 3, takes_mut);
    apply(@mut 3, takes_const);
    apply(@mut 3, takes_imm); //! ERROR (values differ in mutability)
}
