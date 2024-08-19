//@ check-pass

fn mutable_address_of() {
    let mut x = 0;
    let y = &raw mut x;
}

fn mutable_address_of_closure() {
    let mut x = 0;
    let mut f = || {
        let y = &raw mut x;
    };
    f();
}

fn const_address_of_closure() {
    let x = 0;
    let f = || {
        let y = &raw const x;
    };
    f();
}

fn make_fn<F: Fn()>(f: F) -> F { f }

fn const_address_of_fn_closure() {
    let x = 0;
    let f = make_fn(|| {
        let y = &raw const x;
    });
    f();
}

fn const_address_of_fn_closure_move() {
    let x = 0;
    let f = make_fn(move || {
        let y = &raw const x;
    });
    f();
}

fn main() {}
