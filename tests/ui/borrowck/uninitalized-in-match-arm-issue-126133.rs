enum E {
    A,
    B,
    C,
}

fn foo(e: E) {
    let bar;

    match e {
        E::A if true => return,
        E::A => return,
        E::B => {}
        E::C => {
            bar = 5;
        }
    }

    let _baz = bar; //~ ERROR E0381
}

fn main() {}
