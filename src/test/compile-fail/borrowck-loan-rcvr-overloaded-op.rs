// xfail-fast  (compile-flags unsupported on windows)
// compile-flags:--borrowck=err

type point = { x: int, y: int };

impl foo for point {
    pure fn +(z: int) -> int { self.x + self.y + z }
    fn *(z: int) -> int { self.x * self.y * z }
}

fn a() {
    let mut p = {x: 3, y: 4};

    // ok (we can loan out rcvr)
    p + 3;
    p * 3;
}

fn b() {
    let mut p = {x: 3, y: 4};

    // Here I create an outstanding loan and check that we get conflicts:

    &mut p; //! NOTE prior loan as mutable granted here
    //!^ NOTE prior loan as mutable granted here

    p + 3; //! ERROR loan of mutable local variable as immutable conflicts with prior loan
    p * 3; //! ERROR loan of mutable local variable as immutable conflicts with prior loan
}

fn c() {
    // Here the receiver is in aliased memory and hence we cannot
    // consider it immutable:
    let q = @mut {x: 3, y: 4};

    // ...this is ok for pure fns
    *q + 3;


    // ...but not impure fns
    *q * 3; //! ERROR illegal borrow unless pure: creating immutable alias to aliasable, mutable memory
    //!^ NOTE impure due to access to impure function
}

fn main() {
}

