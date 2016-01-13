#![feature(plugin)]

#![plugin(clippy)]
#![deny(clippy)]
#![allow(unused)]

use std::borrow::Cow;

enum Foo { Bar, Baz(u8) }
use Foo::*;

fn single_match(){
    let x = Some(1u8);

    match x {  //~ ERROR you seem to be trying to use match
               //~^ HELP try
        Some(y) => {
            println!("{:?}", y);
        }
        _ => ()
    }

    let z = (1u8,1u8);
    match z { //~ ERROR you seem to be trying to use match
              //~^ HELP try
        (2...3, 7...9) => println!("{:?}", z),
        _ => {}
    }

    // Not linted (pattern guards used)
    match x {
        Some(y) if y == 0 => println!("{:?}", y),
        _ => ()
    }

    // Not linted (content in the else)
    match z {
        (2...3, 7...9) => println!("{:?}", z),
        _ => println!("nope"),
    }
}

fn single_match_know_enum() {
    let x = Some(1u8);
    let y : Result<_, i8> = Ok(1i8);

    match x { //~ ERROR you seem to be trying to use match
              //~^ HELP try
        Some(y) => println!("{:?}", y),
        None => ()
    }

    match y { //~ ERROR you seem to be trying to use match
              //~^ HELP try
        Ok(y) => println!("{:?}", y),
        Err(..) => ()
    }

    let c = Cow::Borrowed("");

    match c { //~ ERROR you seem to be trying to use match
              //~^ HELP try
        Cow::Borrowed(..) => println!("42"),
        Cow::Owned(..) => (),
    }

    let z = Foo::Bar;
    // no warning
    match z {
        Bar => println!("42"),
        Baz(_) => (),
    }

    match z {
        Baz(_) => println!("42"),
        Bar => (),
    }
}

fn match_bool() {
    let test: bool = true;

    match test {  //~ ERROR you seem to be trying to match on a boolean expression
        true => (),
        false => (),
    };

    let option = 1;
    match option == 1 {  //~ ERROR you seem to be trying to match on a boolean expression
        true => 1,
        false => 0,
    };

    match test { //~ ERROR you seem to be trying to match on a boolean expression
        true => (),
        false => { println!("Noooo!"); }
    };

    match test { //~ ERROR you seem to be trying to match on a boolean expression
        false => { println!("Noooo!"); }
        _ => (),
    };

    match test { //~ ERROR you seem to be trying to match on a boolean expression
        false => { println!("Noooo!"); }
        true => { println!("Yes!"); }
    };

    // Not linted
    match option {
        1 ... 10 => (),
        11 ... 20 => (),
        _ => (),
    };
}

fn ref_pats() {
    {
        let v = &Some(0);
        match v {  //~ERROR dereference the expression: `match *v { ...`
            &Some(v) => println!("{:?}", v),
            &None => println!("none"),
        }
        match v {  // this doesn't trigger, we have a different pattern
            &Some(v) => println!("some"),
            other => println!("other"),
        }
    }
    let tup =& (1, 2);
    match tup {  //~ERROR dereference the expression: `match *tup { ...`
        &(v, 1) => println!("{}", v),
        _ => println!("none"),
    }
    // special case: using & both in expr and pats
    let w = Some(0);
    match &w {  //~ERROR use `match w { ...`
        &Some(v) => println!("{:?}", v),
        &None => println!("none"),
    }
    // false positive: only wildcard pattern
    let w = Some(0);
    match w {
        _ => println!("none"),
    }

    let a = &Some(0);
    if let &None = a { //~ERROR dereference the expression: `if let ... = *a {`
        println!("none");
    }

    let b = Some(0);
    if let &None = &b { //~ERROR use `if let ... = b {`
        println!("none");
    }
}

fn overlapping() {
    const FOO : u64 = 2;

    match 42 {
        0 ... 10 => println!("0 ... 10"), //~ERROR: some ranges overlap
        0 ... 11 => println!("0 ... 10"),
        _ => (),
    }

    match 42 {
        0 ... 5 => println!("0 ... 5"), //~ERROR: some ranges overlap
        6 ... 7 => println!("6 ... 7"),
        FOO ... 11 => println!("0 ... 10"),
        _ => (),
    }

    match 42 {
        2 => println!("2"),
        0 ... 5 => println!("0 ... 5"), //~ERROR: some ranges overlap
        _ => (),
    }

    match 42 {
        0 ... 10 => println!("0 ... 10"),
        11 ... 50 => println!("0 ... 10"),
        _ => (),
    }
}

fn main() {
}
