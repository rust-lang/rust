//@ run-pass

#![feature(never_type, never_type_fallback)]
#![feature(exhaustive_patterns)]

#![allow(unreachable_patterns)]
#![allow(unreachable_code)]
#![allow(unused_variables)]

#[allow(dead_code)]
fn foo(z: !) {
    let x: Result<!, !> = Ok(z);

    let Ok(_y) = x;
    let Err(_y) = x;

    let x = [z; 1];

    match x {};
    match x {
        [q] => q,
    };
}

fn bar(nevers: &[!]) {
    match nevers {
        &[]  => (),
    };

    match nevers {
        &[]  => (),
        &[_]  => (),
        &[_, _, _, ..]  => (),
    };
}

fn main() {
    let x: Result<u32, !> = Ok(123);
    let Ok(y) = x;

    assert_eq!(123, y);

    match x {
        Ok(y) => y,
    };

    match x {
        Ok(y) => y,
        Err(e) => match e {},
    };

    let x: Result<u32, &!> = Ok(123);
    match x {
        Ok(y) => y,
        Err(_) => unimplemented!(),
    };

    bar(&[]);
}
