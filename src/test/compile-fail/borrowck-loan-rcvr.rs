type point = { x: int, y: int };

impl foo for point {
    fn impurem() {
    }

    fn blockm(f: fn()) { f() }

    pure fn purem() {
    }
}

fn a() {
    let mut p = {x: 3, y: 4};

    // Here: it's ok to call even though receiver is mutable, because we
    // can loan it out.
    p.purem();
    p.impurem();

    // But in this case we do not honor the loan:
    do p.blockm || { //! NOTE loan of mutable local variable granted here
        p.x = 10; //! ERROR assigning to mutable field prohibited due to outstanding loan
    }
}

fn b() {
    let mut p = {x: 3, y: 4};

    // Here I create an outstanding loan and check that we get conflicts:

    &mut p; //! NOTE prior loan as mutable granted here
    //!^ NOTE prior loan as mutable granted here

    p.purem(); //! ERROR loan of mutable local variable as immutable conflicts with prior loan
    p.impurem(); //! ERROR loan of mutable local variable as immutable conflicts with prior loan
}

fn c() {
    // Here the receiver is in aliased memory and hence we cannot
    // consider it immutable:
    let q = @mut {x: 3, y: 4};

    // ...this is ok for pure fns
    (*q).purem();

    // ...but not impure fns
    (*q).impurem(); //! ERROR illegal borrow unless pure: creating immutable alias to aliasable, mutable memory
    //!^ NOTE impure due to access to impure function
}

fn main() {
}

