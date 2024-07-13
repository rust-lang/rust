fn mutable_address_of() {
    let x = 0;
    let y = &raw mut x;                 //~ ERROR cannot borrow
}

fn mutable_address_of_closure() {
    let x = 0;
    let mut f = || {
        let y = &raw mut x;             //~ ERROR cannot borrow
    };
    f();
}

fn mutable_address_of_imm_closure() {
    let mut x = 0;
    let f = || {
        let y = &raw mut x;
    };
    f();                                //~ ERROR cannot borrow
}

fn make_fn<F: Fn()>(f: F) -> F { f }

fn mutable_address_of_fn_closure() {
    let mut x = 0;
    let f = make_fn(|| {
        let y = &raw mut x;             //~ ERROR cannot borrow
    });
    f();
}

fn mutable_address_of_fn_closure_move() {
    let mut x = 0;
    let f = make_fn(move || {
        let y = &raw mut x;             //~ ERROR cannot borrow
    });
    f();
}

fn main() {}
