// Check that closure captures for slice patterns are inferred correctly

#![allow(unused_variables)]
#![allow(dropping_references)]

//@ run-pass

fn arr_by_ref(x: [String; 3]) {
    let r = &x;
    let f = || {
        let [ref y, ref z @ ..] = x;
    };
    f();
    f();
    // Ensure `x` was borrowed
    drop(r);
    // Ensure that `x` wasn't moved from.
    drop(x);
}

fn arr_by_mut(mut x: [String; 3]) {
    let mut f = || {
        let [ref mut y, ref mut z @ ..] = x;
    };
    f();
    f();
    drop(x);
}

fn arr_by_move(x: [String; 3]) {
    let f = || {
        let [y, z @ ..] = x;
    };
    f();
}

fn arr_ref_by_ref(x: &[String; 3]) {
    let r = &x;
    let f = || {
        let [ref y, ref z @ ..] = *x;
    };
    let g = || {
        let [y, z @ ..] = x;
    };
    f();
    g();
    f();
    g();
    drop(r);
    drop(x);
}

fn arr_ref_by_mut(x: &mut [String; 3]) {
    let mut f = || {
        let [ref mut y, ref mut z @ ..] = *x;
    };
    f();
    f();
    let mut g = || {
        let [y, z @ ..] = x;
        // Ensure binding mode was chosen correctly:
        std::mem::swap(y, &mut z[0]);
    };
    g();
    g();
    drop(x);
}

fn arr_box_by_move(x: Box<[String; 3]>) {
    let f = || {
        let [y, z @ ..] = *x;
    };
    f();
}

fn slice_by_ref(x: &[String]) {
    let r = &x;
    let f = || {
        if let [ref y, ref z @ ..] = *x {}
    };
    let g = || {
        if let [y, z @ ..] = x {}
    };
    f();
    g();
    f();
    g();
    drop(r);
    drop(x);
}

fn slice_by_mut(x: &mut [String]) {
    let mut f = || {
        if let [ref mut y, ref mut z @ ..] = *x {}
    };
    f();
    f();
    let mut g = || {
        if let [y, z @ ..] = x {
            // Ensure binding mode was chosen correctly:
            std::mem::swap(y, &mut z[0]);
        }
    };
    g();
    g();
    drop(x);
}

fn main() {
    arr_by_ref(Default::default());
    arr_by_mut(Default::default());
    arr_by_move(Default::default());
    arr_ref_by_ref(&Default::default());
    arr_ref_by_mut(&mut Default::default());
    arr_box_by_move(Default::default());
    slice_by_ref(&<[_; 3]>::default());
    slice_by_mut(&mut <[_; 3]>::default());
}
