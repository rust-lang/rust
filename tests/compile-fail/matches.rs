#![feature(plugin)]

#![plugin(clippy)]
#![deny(clippy)]
#![allow(unused)]
#![deny(single_match_else)]

use std::borrow::Cow;

enum Foo { Bar, Baz(u8) }
use Foo::*;

enum ExprNode {
    ExprAddrOf,
    Butterflies,
    Unicorns,
}

static NODE: ExprNode = ExprNode::Unicorns;

fn dummy() {
}

fn unwrap_addr() -> Option<&'static ExprNode> {
    match ExprNode::Butterflies {
        //~^ ERROR you seem to be trying to use match
        //~| HELP try
        //~| SUGGESTION if let ExprNode::ExprAddrOf = ExprNode::Butterflies { Some(&NODE) } else { let x = 5; None }
        ExprNode::ExprAddrOf => Some(&NODE),
        _ => { let x = 5; None },
    }
}

fn single_match(){
    let x = Some(1u8);

    match x {
        //~^ ERROR you seem to be trying to use match
        //~| HELP try
        //~| SUGGESTION if let Some(y) = x { println!("{:?}", y); };
        Some(y) => { println!("{:?}", y); }
        _ => ()
    };

    let z = (1u8,1u8);
    match z {
        //~^ ERROR you seem to be trying to use match
        //~| HELP try
        //~| SUGGESTION if let (2...3, 7...9) = z { dummy() };
        (2...3, 7...9) => dummy(),
        _ => {}
    };

    // Not linted (pattern guards used)
    match x {
        Some(y) if y == 0 => println!("{:?}", y),
        _ => ()
    }

    // Not linted (no block with statements in the single arm)
    match z {
        (2...3, 7...9) => println!("{:?}", z),
        _ => println!("nope"),
    }
}

fn single_match_know_enum() {
    let x = Some(1u8);
    let y : Result<_, i8> = Ok(1i8);

    match x {
        //~^ ERROR you seem to be trying to use match
        //~| HELP try
        //~| SUGGESTION if let Some(y) = x { dummy() };
        Some(y) => dummy(),
        None => ()
    };

    match y {
        //~^ ERROR you seem to be trying to use match
        //~| HELP try
        //~| SUGGESTION if let Ok(y) = y { dummy() };
        Ok(y) => dummy(),
        Err(..) => ()
    };

    let c = Cow::Borrowed("");

    match c {
        //~^ ERROR you seem to be trying to use match
        //~| HELP try
        //~| SUGGESTION if let Cow::Borrowed(..) = c { dummy() };
        Cow::Borrowed(..) => dummy(),
        Cow::Owned(..) => (),
    };

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

    match test {
    //~^ ERROR you seem to be trying to match on a boolean expression
    //~| HELP try
    //~| SUGGESTION if test { 0 } else { 42 };
        true => 0,
        false => 42,
    };

    let option = 1;
    match option == 1 {
    //~^ ERROR you seem to be trying to match on a boolean expression
    //~| HELP try
    //~| SUGGESTION if option == 1 { 1 } else { 0 };
        true => 1,
        false => 0,
    };

    match test {
    //~^ ERROR you seem to be trying to match on a boolean expression
    //~| HELP try
    //~^^ SUGGESTION if !test { println!("Noooo!"); };
        true => (),
        false => { println!("Noooo!"); }
    };

    match test {
    //~^ ERROR you seem to be trying to match on a boolean expression
    //~| HELP try
    //~^^ SUGGESTION if !test { println!("Noooo!"); };
        false => { println!("Noooo!"); }
        _ => (),
    };

    match test {
    //~^ ERROR you seem to be trying to match on a boolean expression
    //~| HELP try
    //~| SUGGESTION if test { println!("Yes!"); } else { println!("Noooo!"); };
        false => { println!("Noooo!"); }
        true => { println!("Yes!"); }
    };

    // Not linted
    match option {
        1 ... 10 => 1,
        11 ... 20 => 2,
        _ => 3,
    };
}

fn ref_pats() {
    {
        let v = &Some(0);
        match v {
            //~^ERROR add `&` to all patterns
            //~|HELP instead of
            //~|SUGGESTION match *v { .. }
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
        //~^ERROR add `&` to all patterns
        //~|HELP instead of
        //~|SUGGESTION match *tup { .. }
        &(v, 1) => println!("{}", v),
        _ => println!("none"),
    }
    // special case: using & both in expr and pats
    let w = Some(0);
    match &w {
        //~^ERROR add `&` to both
        //~|HELP try
        //~|SUGGESTION match w { .. }
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
        //~^ERROR add `&` to all patterns
        //~|HELP instead of
        //~|SUGGESTION if let .. = *a { .. }
        println!("none");
    }

    let b = Some(0);
    if let &None = &b {
        //~^ERROR add `&` to both
        //~|HELP try
        //~|SUGGESTION if let .. = b { .. }
        println!("none");
    }
}

fn overlapping() {
    const FOO : u64 = 2;

    match 42 {
        0 ... 10 => println!("0 ... 10"), //~ERROR: some ranges overlap
        0 ... 11 => println!("0 ... 10"), //~NOTE overlaps with this
        _ => (),
    }

    match 42 {
        0 ... 5 => println!("0 ... 5"), //~ERROR: some ranges overlap
        6 ... 7 => println!("6 ... 7"),
        FOO ... 11 => println!("0 ... 10"), //~NOTE overlaps with this
        _ => (),
    }

    match 42 {
        2 => println!("2"), //~NOTE overlaps with this
        0 ... 5 => println!("0 ... 5"), //~ERROR: some ranges overlap
        _ => (),
    }

    match 42 {
        0 ... 10 => println!("0 ... 10"),
        11 ... 50 => println!("0 ... 10"),
        _ => (),
    }

    if let None = Some(42) {
        // nothing
    } else if let None = Some(42) {
        // another nothing :-)
    }
}

fn main() {
}
