fn main() {}

struct U;

fn slice() {
    let mut arr = [U, U, U, U, U];
    let hold_all = &arr;
    let [ref _x0_hold, _x1, ref xs_hold @ ..] = arr; //~ ERROR cannot move out of `arr[..]`
    _x1 = U; //~ ERROR cannot assign twice to immutable variable `_x1`
    drop(hold_all);
    let [_x0, ..] = arr; //~ ERROR cannot move out of `arr[..]`
    drop(_x0_hold);
    let [_, _, ref mut _x2, _x3, mut _x4] = arr;
    //~^ ERROR cannot borrow `arr[..]` as mutable
    //~| ERROR cannot move out of `arr[..]` because it is borrowed
    //~| ERROR cannot move out of `arr[..]` because it is borrowed
    drop(xs_hold);
}

fn tuple() {
    let mut tup = (U, U, U, U);
    let (ref _x0, _x1, ref _x2, ..) = tup;
    _x1 = U; //~ ERROR cannot assign twice to immutable variable
    let _x0_hold = &mut tup.0; //~ ERROR cannot borrow `tup.0` as mutable because it is also
    let (ref mut _x0_hold, ..) = tup; //~ ERROR cannot borrow `tup.0` as mutable because it is also
    *_x0 = U; //~ ERROR cannot assign to `*_x0`, which is behind a `&` reference
    *_x2 = U; //~ ERROR cannot assign to `*_x2`, which is behind a `&` reference
    drop(tup.1); //~ ERROR use of moved value: `tup.1`
    let _x1_hold = &tup.1; //~ ERROR borrow of moved value: `tup.1`
    let (.., ref mut _x3) = tup;
    let _x3_hold = &tup.3; //~ ERROR cannot borrow `tup.3` as immutable
    let _x3_hold = &mut tup.3; //~ ERROR cannot borrow `tup.3` as mutable more
    let (.., ref mut _x4_hold) = tup; //~ ERROR cannot borrow `tup.3` as mutable more
    let (.., ref _x4_hold) = tup; //~ ERROR cannot borrow `tup.3` as immutable
    drop(_x3);
}

fn closure() {
    let mut tup = (U, U, U);
    let c1 = || {
        let (ref _x0, _x1, _) = tup;
    };
    let c2 = || {
        //~^ ERROR use of moved value
        let (ref mut _x0, _, _x2) = tup;
    };
    drop(c1);
}
