// xfail-fast  (compile-flags unsupported on windows)
// compile-flags:--borrowck=err

type point = { x: int, y: int };

impl foo for point {
    fn impurem() {
    }

    pure fn purem() {
    }
}

fn a() {
    let mut p = {x: 3, y: 4};
    p.purem();
    p.impurem();
}

fn a2() {
    let mut p = {x: 3, y: 4};
    p.purem();
    p.impurem();
    p.x = p.y;
}

fn b() {
    let mut p = {x: 3, y: 4};

    &mut p; //! NOTE prior loan as mutable granted here
    //!^ NOTE prior loan as mutable granted here

    p.purem(); //! ERROR loan of mutable local variable as immutable conflicts with prior loan
    p.impurem(); //! ERROR loan of mutable local variable as immutable conflicts with prior loan
}

fn c() {
    let q = @mut {x: 3, y: 4};
    (*q).purem();
    (*q).impurem(); //! ERROR illegal borrow unless pure: creating immutable alias to aliasable, mutable memory
    //!^ NOTE impure due to access to impure function
}

fn main() {
}

