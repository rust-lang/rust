#![feature(slice_patterns)]
#![deny(unreachable_patterns)]

fn main() {
    let s: &[bool] = &[true; 0];
    let s0: &[bool; 0] = &[];
    let s1: &[bool; 1] = &[false; 1];
    let s2: &[bool; 2] = &[false; 2];
    let s3: &[bool; 3] = &[false; 3];

    let [] = s0;
    let [_] = s1;
    let [_, _] = s2;

    let [..] = s;
    let [..] = s0;
    let [..] = s1;
    let [..] = s2;
    let [..] = s3;

    let [_, _, ..] = s2;
    let [_, .., _] = s2;
    let [.., _, _] = s2;

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
    //~^ ERROR `&[false, _, true]` not covered
        [true, ..] => {}
        [.., false] => {}
    }
    match s {
    //~^ ERROR `&[false, true]` not covered
        [] => {}
        [true, ..] => {}
        [.., false] => {}
    }

    match s3 {
    //~^ ERROR `&[false, _, _]` not covered
        [true, .., true] => {}
    }
    match s {
    //~^ ERROR `&[_, ..]` not covered
        [] => {}
    }
    match s {
    //~^ ERROR `&[_, _, ..]` not covered
        [] => {}
        [_] => {}
    }
    match s {
    //~^ ERROR `&[false]` not covered
        [] => {}
        [true, ..] => {}
    }
    match s {
    //~^ ERROR `&[false, _]` not covered
        [] => {}
        [_] => {}
        [true, ..] => {}
    }
    match s {
    //~^ ERROR `&[_, false]` not covered
        [] => {}
        [_] => {}
        [.., true] => {}
    }

    match s {
        [true, ..] => {}
        [true, ..] => {} //~ ERROR unreachable pattern
        [true] => {} //~ ERROR unreachable pattern
        [..] => {}
    }
    match s {
        [.., true] => {}
        [.., true] => {} //~ ERROR unreachable pattern
        [true] => {} //~ ERROR unreachable pattern
        [..] => {}
    }
    match s {
        [false, .., true] => {}
        [false, .., true] => {} //~ ERROR unreachable pattern
        [false, true] => {} //~ ERROR unreachable pattern
        [false] => {}
        [..] => {}
    }
    match s {
    //~^ ERROR `&[_, _, true]` not covered
        [] => {}
        [_] => {}
        [_, _] => {}
        [.., false] => {}
    }
    match s {
    //~^ ERROR `&[true, _, _]` not covered
        [] => {}
        [_] => {}
        [_, _] => {}
        [false, .., false] => {}
    }
}
