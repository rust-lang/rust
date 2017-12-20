
#![feature(exclusive_range_pattern)]


#![warn(clippy)]
#![allow(unused, if_let_redundant_pattern_matching)]
#![warn(single_match_else)]

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
        ExprNode::ExprAddrOf => Some(&NODE),
        _ => { let x = 5; None },
    }
}

fn single_match(){
    let x = Some(1u8);

    match x {
        Some(y) => { println!("{:?}", y); }
        _ => ()
    };

    let z = (1u8,1u8);
    match z {
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
        Some(y) => dummy(),
        None => ()
    };

    match y {
        Ok(y) => dummy(),
        Err(..) => ()
    };

    let c = Cow::Borrowed("");

    match c {
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
        true => 0,
        false => 42,
    };

    let option = 1;
    match option == 1 {
        true => 1,
        false => 0,
    };

    match test {
        true => (),
        false => { println!("Noooo!"); }
    };

    match test {
        false => { println!("Noooo!"); }
        _ => (),
    };

    match test && test {
        false => { println!("Noooo!"); }
        _ => (),
    };

    match test {
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

fn overlapping() {
    const FOO : u64 = 2;

    match 42 {
        0 ... 10 => println!("0 ... 10"),
        0 ... 11 => println!("0 ... 11"),
        _ => (),
    }

    match 42 {
        0 ... 5 => println!("0 ... 5"),
        6 ... 7 => println!("6 ... 7"),
        FOO ... 11 => println!("0 ... 11"),
        _ => (),
    }

    match 42 {
        2 => println!("2"),
        0 ... 5 => println!("0 ... 5"),
        _ => (),
    }

    match 42 {
        2 => println!("2"),
        0 ... 2 => println!("0 ... 2"),
        _ => (),
    }

    match 42 {
        0 ... 10 => println!("0 ... 10"),
        11 ... 50 => println!("11 ... 50"),
        _ => (),
    }

    match 42 {
        2 => println!("2"),
        0 .. 2 => println!("0 .. 2"),
        _ => (),
    }

    match 42 {
        0 .. 10 => println!("0 .. 10"),
        10 .. 50 => println!("10 .. 50"),
        _ => (),
    }

    match 42 {
        0 .. 11 => println!("0 .. 11"),
        0 ... 11 => println!("0 ... 11"),
        _ => (),
    }

    if let None = Some(42) {
        // nothing
    } else if let None = Some(42) {
        // another nothing :-)
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
