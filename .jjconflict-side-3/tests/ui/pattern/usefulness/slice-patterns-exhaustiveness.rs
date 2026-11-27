fn main() {
    let s: &[bool] = &[true; 0];
    let s1: &[bool; 1] = &[false; 1];
    let s2: &[bool; 2] = &[false; 2];
    let s3: &[bool; 3] = &[false; 3];
    let s10: &[bool; 10] = &[false; 10];

    match s2 {
        //~^ ERROR `&[false, _]` not covered
        [true, .., true] => {}
    }
    match s3 {
        //~^ ERROR `&[false, ..]` not covered
        [true, .., true] => {}
    }
    match s10 {
        //~^ ERROR `&[false, ..]` not covered
        [true, .., true] => {}
    }

    match s1 {
        [true, ..] => {}
        [.., false] => {}
    }
    match s2 {
        //~^ ERROR `&[false, true]` not covered
        [true, ..] => {}
        [.., false] => {}
    }
    match s3 {
        //~^ ERROR `&[false, .., true]` not covered
        [true, ..] => {}
        [.., false] => {}
    }
    match s {
        //~^ ERROR `&[false, .., true]` not covered
        [] => {}
        [true, ..] => {}
        [.., false] => {}
    }

    match s {
        //~^ ERROR `&[_, ..]` not covered
        [] => {}
    }
    match s {
        //~^ ERROR `&[]` and `&[_, ..]` not covered
        [..] if false => {}
    }
    match s {
        //~^ ERROR `&[_, _, ..]` not covered
        [] => {}
        [_] => {}
    }
    match s {
        //~^ ERROR `&[false, ..]` not covered
        [] => {}
        [true, ..] => {}
    }
    match s {
        //~^ ERROR `&[false, _, ..]` not covered
        [] => {}
        [_] => {}
        [true, ..] => {}
    }
    match s {
        //~^ ERROR `&[_, .., false]` not covered
        [] => {}
        [_] => {}
        [.., true] => {}
    }

    match s {
        //~^ ERROR `&[_, _, .., true]` not covered
        [] => {}
        [_] => {}
        [_, _] => {}
        [.., false] => {}
    }
    match s {
        //~^ ERROR `&[true, _, .., _]` not covered
        [] => {}
        [_] => {}
        [_, _] => {}
        [false, .., false] => {}
    }

    const CONST: &[bool] = &[true];
    match s {
        //~^ ERROR `&[]` and `&[_, _, ..]` not covered
        &[true] => {}
    }
    match s {
        //~^ ERROR `&[]` and `&[_, _, ..]` not covered
        CONST => {}
    }
    match s {
        //~^ ERROR `&[]` and `&[_, _, ..]` not covered
        CONST => {}
        &[false] => {}
    }
    match s {
        //~^ ERROR `&[]` and `&[_, _, ..]` not covered
        &[false] => {}
        CONST => {}
    }
    match s {
        //~^ ERROR `&[_, _, ..]` not covered
        &[] => {}
        CONST => {}
    }
    match s {
        //~^ ERROR `&[false]` not covered
        &[] => {}
        CONST => {}
        &[_, _, ..] => {}
    }
    match s {
        [] => {}
        [false] => {}
        CONST => {}
        [_, _, ..] => {}
    }
    const CONST1: &[bool; 1] = &[true];
    match s1 {
        //~^ ERROR `&[false]` not covered
        CONST1 => {}
    }
    match s1 {
        CONST1 => {}
        [false] => {}
    }
}
