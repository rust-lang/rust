// xfail-fast  (compile-flags unsupported on windows)
// compile-flags:--borrowck=err

type point = { x: int, y: int };

fn a() {
    let mut p = {x: 3, y: 4};
    let _q = &p; //! NOTE loan of mutable local variable granted here

    // This assignment is illegal because the field x is not
    // inherently mutable; since `p` was made immutable, `p.x` is now
    // immutable.  Otherwise the type of &_q.x (&int) would be wrong.
    p.x = 5; //! ERROR assigning to mutable field prohibited due to outstanding loan
}

fn b() {
    let mut p = {x: 3, mut y: 4};
    let _q = &p;

    // This assignment is legal because `y` is inherently mutable (and
    // hence &_q.y is &mut int).
    p.y = 5;
}

fn c() {
    // this is sort of the opposite.  We take a loan to the interior of `p`
    // and then try to overwrite `p` as a whole.

    let mut p = {x: 3, mut y: 4};
    let _q = &p.y; //! NOTE loan of mutable local variable granted here
    p = {x: 5, mut y: 7};//! ERROR assigning to mutable local variable prohibited due to outstanding loan
    copy p;
}

fn d() {
    // just for completeness's sake, the easy case, where we take the
    // address of a subcomponent and then modify that subcomponent:

    let mut p = {x: 3, mut y: 4};
    let _q = &p.y; //! NOTE loan of mutable field granted here
    p.y = 5; //! ERROR assigning to mutable field prohibited due to outstanding loan
    copy p;
}

fn main() {
}

