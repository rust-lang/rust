// run-rustfix

#![warn(clippy::manual_split_once)]
#![allow(unused, clippy::iter_skip_next, clippy::iter_nth_zero)]

extern crate itertools;

#[allow(unused_imports)]
use itertools::Itertools;

fn main() {
    let _ = "key=value".splitn(2, '=').nth(2);
    let _ = "key=value".splitn(2, '=').nth(1).unwrap();
    let _ = "key=value".splitn(2, '=').skip(1).next().unwrap();
    let (_, _) = "key=value".splitn(2, '=').next_tuple().unwrap();

    let s = String::from("key=value");
    let _ = s.splitn(2, '=').nth(1).unwrap();

    let s = Box::<str>::from("key=value");
    let _ = s.splitn(2, '=').nth(1).unwrap();

    let s = &"key=value";
    let _ = s.splitn(2, '=').skip(1).next().unwrap();

    fn _f(s: &str) -> Option<&str> {
        let _ = s.splitn(2, '=').nth(1)?;
        let _ = s.splitn(2, '=').skip(1).next()?;
        let _ = s.rsplitn(2, '=').nth(1)?;
        let _ = s.rsplitn(2, '=').skip(1).next()?;
        None
    }

    // Don't lint, slices don't have `split_once`
    let _ = [0, 1, 2].splitn(2, |&x| x == 1).nth(1).unwrap();

    // `rsplitn` gives the results in the reverse order of `rsplit_once`
    let _ = "key=value".rsplitn(2, '=').nth(1).unwrap();
    let (_, _) = "key=value".rsplitn(2, '=').next_tuple().unwrap();
    let _ = s.rsplitn(2, '=').nth(1);
}

fn indirect() -> Option<()> {
    let mut iter = "a.b.c".splitn(2, '.');
    let l = iter.next().unwrap();
    let r = iter.next().unwrap();

    let mut iter = "a.b.c".splitn(2, '.');
    let l = iter.next()?;
    let r = iter.next()?;

    let mut iter = "a.b.c".rsplitn(2, '.');
    let r = iter.next().unwrap();
    let l = iter.next().unwrap();

    let mut iter = "a.b.c".rsplitn(2, '.');
    let r = iter.next()?;
    let l = iter.next()?;

    // could lint, currently doesn't

    let mut iter = "a.b.c".splitn(2, '.');
    let other = 1;
    let l = iter.next()?;
    let r = iter.next()?;

    let mut iter = "a.b.c".splitn(2, '.');
    let mut mut_binding = iter.next()?;
    let r = iter.next()?;

    let mut iter = "a.b.c".splitn(2, '.');
    let tuple = (iter.next()?, iter.next()?);

    // should not lint

    let mut missing_unwrap = "a.b.c".splitn(2, '.');
    let l = missing_unwrap.next();
    let r = missing_unwrap.next();

    let mut mixed_unrap = "a.b.c".splitn(2, '.');
    let unwrap = mixed_unrap.next().unwrap();
    let question_mark = mixed_unrap.next()?;

    let mut iter = "a.b.c".splitn(2, '.');
    let same_name = iter.next()?;
    let same_name = iter.next()?;

    let mut iter = "a.b.c".splitn(2, '.');
    let shadows_existing = "d";
    let shadows_existing = iter.next()?;
    let r = iter.next()?;

    let mut iter = "a.b.c".splitn(2, '.');
    let becomes_shadowed = iter.next()?;
    let becomes_shadowed = "d";
    let r = iter.next()?;

    let mut iter = "a.b.c".splitn(2, '.');
    let l = iter.next()?;
    let r = iter.next()?;
    let third_usage = iter.next()?;

    let mut n_three = "a.b.c".splitn(3, '.');
    let l = n_three.next()?;
    let r = n_three.next()?;

    let mut iter = "a.b.c".splitn(2, '.');
    {
        let in_block = iter.next()?;
    }
    let r = iter.next()?;

    let mut lacks_binding = "a.b.c".splitn(2, '.');
    let _ = lacks_binding.next()?;
    let r = lacks_binding.next()?;

    let mut mapped = "a.b.c".splitn(2, '.').map(|_| "~");
    let l = iter.next()?;
    let r = iter.next()?;

    let mut assigned = "";
    let mut iter = "a.b.c".splitn(2, '.');
    let l = iter.next()?;
    assigned = iter.next()?;

    None
}

#[clippy::msrv = "1.51"]
fn _msrv_1_51() {
    // `str::split_once` was stabilized in 1.52. Do not lint this
    let _ = "key=value".splitn(2, '=').nth(1).unwrap();

    let mut iter = "a.b.c".splitn(2, '.');
    let a = iter.next().unwrap();
    let b = iter.next().unwrap();
}

#[clippy::msrv = "1.52"]
fn _msrv_1_52() {
    let _ = "key=value".splitn(2, '=').nth(1).unwrap();

    let mut iter = "a.b.c".splitn(2, '.');
    let a = iter.next().unwrap();
    let b = iter.next().unwrap();
}
