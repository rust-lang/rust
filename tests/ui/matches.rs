// Copyright 2014-2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.



#![feature(exclusive_range_pattern)]


#![warn(clippy::all)]
#![allow(unused, clippy::redundant_pattern_matching)]
#![warn(clippy::match_same_arms)]


fn dummy() {
}

fn ref_pats() {
    {
        let v = &Some(0);
        match v {
            &Some(v) => println!("{:?}", v),
            &None => println!("none"),
        }
        match v {  // this doesn't trigger, we have a different pattern
            &Some(v) => println!("some"),
            other => println!("other"),
        }
    }
    let tup =& (1, 2);
    match tup {
        &(v, 1) => println!("{}", v),
        _ => println!("none"),
    }
    // special case: using & both in expr and pats
    let w = Some(0);
    match &w {
        &Some(v) => println!("{:?}", v),
        &None => println!("none"),
    }
    // false positive: only wildcard pattern
    let w = Some(0);
    match w {
        _ => println!("none"),
    }

    let a = &Some(0);
    if let &None = a {
        println!("none");
    }

    let b = Some(0);
    if let &None = &b {
        println!("none");
    }
}

fn match_wild_err_arm() {
    let x: Result<i32, &str> = Ok(3);

    match x {
        Ok(3) => println!("ok"),
        Ok(_) => println!("ok"),
        Err(_) => panic!("err")
    }

    match x {
        Ok(3) => println!("ok"),
        Ok(_) => println!("ok"),
        Err(_) => {panic!()}
    }

    match x {
        Ok(3) => println!("ok"),
        Ok(_) => println!("ok"),
        Err(_) => {panic!();}
    }

    // allowed when not with `panic!` block
    match x {
        Ok(3) => println!("ok"),
        Ok(_) => println!("ok"),
        Err(_) => println!("err")
    }

    // allowed when used with `unreachable!`
    match x {
        Ok(3) => println!("ok"),
        Ok(_) => println!("ok"),
        Err(_) => {unreachable!()}
    }

    match x {
        Ok(3) => println!("ok"),
        Ok(_) => println!("ok"),
        Err(_) => unreachable!()
    }

    match x {
        Ok(3) => println!("ok"),
        Ok(_) => println!("ok"),
        Err(_) => {unreachable!();}
    }

    // no warning because of the guard
    match x {
        Ok(x) if x*x == 64 => println!("ok"),
        Ok(_) => println!("ok"),
        Err(_) => println!("err")
    }

    // this used to be a false positive, see #1996
    match x {
        Ok(3) => println!("ok"),
        Ok(x) if x*x == 64 => println!("ok 64"),
        Ok(_) => println!("ok"),
        Err(_) => println!("err")
    }

    match (x, Some(1i32)) {
        (Ok(x), Some(_)) => println!("ok {}", x),
        (Ok(_), Some(x)) => println!("ok {}", x),
        _ => println!("err")
    }

    // no warning because of the different types for x
    match (x, Some(1.0f64)) {
        (Ok(x), Some(_)) => println!("ok {}", x),
        (Ok(_), Some(x)) => println!("ok {}", x),
        _ => println!("err")
    }

    // because of a bug, no warning was generated for this case before #2251
    match x {
        Ok(_tmp) => println!("ok"),
        Ok(3) => println!("ok"),
        Ok(_) => println!("ok"),
        Err(_) => {unreachable!();}
    }
}

fn match_as_ref() {
    let owned: Option<()> = None;
    let borrowed: Option<&()> = match owned {
        None => None,
        Some(ref v) => Some(v),
    };

    let mut mut_owned: Option<()> = None;
    let borrow_mut: Option<&mut ()> = match mut_owned {
        None => None,
        Some(ref mut v) => Some(v),
    };

}

fn main() {
}
