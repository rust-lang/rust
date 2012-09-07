struct C {
    x: uint,
}

fn C(x: uint) -> C {
    C {
        x: x
    }
}

fn f<T:Copy>(_x: T) {
}

#[deny(non_implicitly_copyable_typarams)]
fn main() {
    f(C(1u));
}
