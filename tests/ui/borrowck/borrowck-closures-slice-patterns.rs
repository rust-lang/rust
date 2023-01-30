// Check that closure captures for slice patterns are inferred correctly

fn arr_by_ref(mut x: [String; 3]) {
    let f = || {
        let [ref y, ref z @ ..] = x;
    };
    let r = &mut x;
    //~^ ERROR cannot borrow
    f();
}

fn arr_by_mut(mut x: [String; 3]) {
    let mut f = || {
        let [ref mut y, ref mut z @ ..] = x;
    };
    let r = &x;
    //~^ ERROR cannot borrow
    f();
}

fn arr_by_move(x: [String; 3]) {
    let f = || {
        let [y, z @ ..] = x;
    };
    &x;
    //~^ ERROR borrow of moved value
}

fn arr_ref_by_ref(x: &mut [String; 3]) {
    let f = || {
        let [ref y, ref z @ ..] = *x;
    };
    let r = &mut *x;
    //~^ ERROR cannot borrow
    f();
}

fn arr_ref_by_uniq(x: &mut [String; 3]) {
    let mut f = || {
        let [ref mut y, ref mut z @ ..] = *x;
    };
    let r = &x;
    //~^ ERROR cannot borrow
    f();
}

fn arr_box_by_move(x: Box<[String; 3]>) {
    let f = || {
        let [y, z @ ..] = *x;
    };
    &x;
    //~^ ERROR borrow of moved value
}

fn slice_by_ref(x: &mut [String]) {
    let f = || {
        if let [ref y, ref z @ ..] = *x {}
    };
    let r = &mut *x;
    //~^ ERROR cannot borrow
    f();
}

fn slice_by_uniq(x: &mut [String]) {
    let mut f = || {
        if let [ref mut y, ref mut z @ ..] = *x {}
    };
    let r = &x;
    //~^ ERROR cannot borrow
    f();
}

fn main() {
    arr_by_ref(Default::default());
    arr_by_mut(Default::default());
    arr_by_move(Default::default());
    arr_ref_by_ref(&mut Default::default());
    arr_ref_by_uniq(&mut Default::default());
    arr_box_by_move(Default::default());
    slice_by_ref(&mut <[_; 3]>::default());
    slice_by_uniq(&mut <[_; 3]>::default());
}
