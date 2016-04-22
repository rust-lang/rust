#![feature(custom_attribute)]
#![allow(dead_code, unused_attributes)]

#[miri_run]
fn one_line_ref() -> i16 {
    *&1
}

#[miri_run]
fn basic_ref() -> i16 {
    let x = &1;
    *x
}

#[miri_run]
fn basic_ref_mut() -> i16 {
    let x = &mut 1;
    *x += 2;
    *x
}

#[miri_run]
fn basic_ref_mut_var() -> i16 {
    let mut a = 1;
    {
        let x = &mut a;
        *x += 2;
    }
    a
}

#[miri_run]
fn tuple_ref_mut() -> (i8, i8) {
    let mut t = (10, 20);
    {
        let x = &mut t.1;
        *x += 2;
    }
    t
}

#[miri_run]
fn match_ref_mut() -> i8 {
    let mut t = (20, 22);
    {
        let mut opt = Some(&mut t);
        match opt {
            Some(&mut (ref mut x, ref mut y)) => *x += *y,
            None => {},
        }
    }
    t.0
}

#[miri_run]
fn dangling_pointer() -> *const i32 {
    let b = Box::new(42);
    &*b as *const i32
}

fn main() {}
