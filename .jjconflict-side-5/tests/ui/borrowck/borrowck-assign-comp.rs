struct Point { x: isize, y: isize }

fn a() {
    let mut p = Point {x: 3, y: 4};
    let q = &p;

    // This assignment is illegal because the field x is not
    // inherently mutable; since `p` was made immutable, `p.x` is now
    // immutable.  Otherwise the type of &_q.x (&isize) would be wrong.
    p.x = 5; //~ ERROR cannot assign to `p.x` because it is borrowed
    q.x;
}

fn c() {
    // this is sort of the opposite.  We take a loan to the interior of `p`
    // and then try to overwrite `p` as a whole.

    let mut p = Point {x: 3, y: 4};
    let q = &p.y;
    p = Point {x: 5, y: 7};//~ ERROR cannot assign to `p` because it is borrowed
    p.x; // silence warning
    *q; // stretch loan
}

fn d() {
    // just for completeness's sake, the easy case, where we take the
    // address of a subcomponent and then modify that subcomponent:

    let mut p = Point {x: 3, y: 4};
    let q = &p.y;
    p.y = 5; //~ ERROR cannot assign to `p.y` because it is borrowed
    *q;
}

fn main() {
}
