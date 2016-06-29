#![feature(plugin, inclusive_range_syntax)]
#![feature(dotdot_in_tuple_patterns)]
#![plugin(clippy)]

#![allow(dead_code, no_effect, unnecessary_operation)]
#![allow(let_and_return)]
#![allow(needless_return)]
#![allow(unused_variables)]
#![allow(cyclomatic_complexity)]
#![allow(blacklisted_name)]
#![allow(collapsible_if)]

fn bar<T>(_: T) {}
fn foo() -> bool { unimplemented!() }

struct Foo {
    bar: u8,
}

#[deny(if_same_then_else)]
#[deny(match_same_arms)]
fn if_same_then_else() -> Result<&'static str, ()> {
    if true {
        Foo { bar: 42 };
        0..10;
        ..;
        0..;
        ..10;
        0...10;
        foo();
    }
    else { //~ERROR this `if` has identical blocks
        Foo { bar: 42 };
        0..10;
        ..;
        0..;
        ..10;
        0...10;
        foo();
    }

    if true {
        Foo { bar: 42 };
    }
    else {
        Foo { bar: 43 };
    }

    if true {
        0..10;
    }
    else {
        0...10;
    }

    if true {
        foo();
        foo();
    }
    else {
        foo();
    }

    let _ = if true {
        foo();
        42
    }
    else { //~ERROR this `if` has identical blocks
        foo();
        42
    };

    if true {
        foo();
    }

    let _ = if true {
        42
    }
    else { //~ERROR this `if` has identical blocks
        42
    };

    if true {
        let bar = if true {
            42
        }
        else {
            43
        };

        while foo() { break; }
        bar + 1;
    }
    else { //~ERROR this `if` has identical blocks
        let bar = if true {
            42
        }
        else {
            43
        };

        while foo() { break; }
        bar + 1;
    }

    if true {
        let _ = match 42 {
            42 => 1,
            a if a > 0 => 2,
            10...15 => 3,
            _ => 4,
        };
    }
    else if false {
        foo();
    }
    else if foo() { //~ERROR this `if` has identical blocks
        let _ = match 42 {
            42 => 1,
            a if a > 0 => 2,
            10...15 => 3,
            _ => 4,
        };
    }

    if true {
        if let Some(a) = Some(42) {}
    }
    else { //~ERROR this `if` has identical blocks
        if let Some(a) = Some(42) {}
    }

    if true {
        if let (1, .., 3) = (1, 2, 3) {}
    }
    else { //~ERROR this `if` has identical blocks
        if let (1, .., 3) = (1, 2, 3) {}
    }

    if true {
        if let (1, .., 3) = (1, 2, 3) {}
    }
    else {
        if let (.., 3) = (1, 2, 3) {}
    }

    if true {
        if let (1, .., 3) = (1, 2, 3) {}
    }
    else {
        if let (.., 4) = (1, 2, 3) {}
    }

    if true {
        if let (1, .., 3) = (1, 2, 3) {}
    }
    else {
        if let (.., 1, 3) = (1, 2, 3) {}
    }

    if true {
        if let Some(a) = Some(42) {}
    }
    else {
        if let Some(a) = Some(43) {}
    }

    let _ = match 42 {
        42 => foo(),
        51 => foo(), //~ERROR this `match` has identical arm bodies
        _ => true,
    };

    let _ = match Some(42) {
        Some(_) => 24,
        None => 24, //~ERROR this `match` has identical arm bodies
    };

    let _ = match Some(42) {
        Some(foo) => 24,
        None => 24,
    };

    let _ = match Some(42) {
        Some(42) => 24,
        Some(a) => 24, // bindings are different
        None => 0,
    };

    let _ = match Some(42) {
        Some(a) if a > 0 => 24,
        Some(a) => 24, // one arm has a guard
        None => 0,
    };

    match (Some(42), Some(42)) {
        (Some(a), None) => bar(a),
        (None, Some(a)) => bar(a), //~ERROR this `match` has identical arm bodies
        _ => (),
    }

    match (Some(42), Some(42)) {
        (Some(a), ..) => bar(a),
        (.., Some(a)) => bar(a), //~ERROR this `match` has identical arm bodies
        _ => (),
    }

    match (1, 2, 3) {
        (1, .., 3) => 42,
        (.., 3) => 42, //~ERROR this `match` has identical arm bodies
        _ => 0,
    };

    match (Some(42), Some("")) {
        (Some(a), None) => bar(a),
        (None, Some(a)) => bar(a), // bindings have different types
        _ => (),
    }

    if true {
        try!(Ok("foo"));
    }
    else { //~ERROR this `if` has identical blocks
        try!(Ok("foo"));
    }

    if true {
        let foo = "";
        return Ok(&foo[0..]);
    }
    else if false {
        let foo = "bar";
        return Ok(&foo[0..]);
    }
    else { //~ERROR this `if` has identical blocks
        let foo = "";
        return Ok(&foo[0..]);
    }
}

#[deny(ifs_same_cond)]
#[allow(if_same_then_else)] // all empty blocks
fn ifs_same_cond() {
    let a = 0;
    let b = false;

    if b {
    }
    else if b { //~ERROR this `if` has the same condition as a previous if
    }

    if a == 1 {
    }
    else if a == 1 { //~ERROR this `if` has the same condition as a previous if
    }

    if 2*a == 1 {
    }
    else if 2*a == 2 {
    }
    else if 2*a == 1 { //~ERROR this `if` has the same condition as a previous if
    }
    else if a == 1 {
    }

    // See #659
    if cfg!(feature = "feature1-659") {
        1
    } else if cfg!(feature = "feature2-659") {
        2
    } else {
        3
    };

    let mut v = vec![1];
    if v.pop() == None { // ok, functions
    }
    else if v.pop() == None {
    }

    if v.len() == 42 { // ok, functions
    }
    else if v.len() == 42 {
    }
}

fn main() {}
