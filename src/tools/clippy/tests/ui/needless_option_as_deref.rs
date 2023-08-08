//@run-rustfix

#![allow(unused)]
#![warn(clippy::needless_option_as_deref)]
#![allow(clippy::useless_vec)]

fn main() {
    // should lint
    let _: Option<&usize> = Some(&1).as_deref();
    let _: Option<&mut usize> = Some(&mut 1).as_deref_mut();

    let mut y = 0;
    let mut x = Some(&mut y);
    let _ = x.as_deref_mut();

    // should not lint
    let _ = Some(Box::new(1)).as_deref();
    let _ = Some(Box::new(1)).as_deref_mut();

    let mut y = 0;
    let mut x = Some(&mut y);
    for _ in 0..3 {
        let _ = x.as_deref_mut();
    }

    let mut y = 0;
    let mut x = Some(&mut y);
    let mut closure = || {
        let _ = x.as_deref_mut();
    };
    closure();
    closure();

    // #7846
    let mut i = 0;
    let mut opt_vec = vec![Some(&mut i)];
    opt_vec[0].as_deref_mut().unwrap();

    let mut i = 0;
    let x = &mut Some(&mut i);
    (*x).as_deref_mut();

    // #8047
    let mut y = 0;
    let mut x = Some(&mut y);
    x.as_deref_mut();
    dbg!(x);
}

struct S<'a> {
    opt: Option<&'a mut usize>,
}

fn from_field<'a>(s: &'a mut S<'a>) -> Option<&'a mut usize> {
    s.opt.as_deref_mut()
}
