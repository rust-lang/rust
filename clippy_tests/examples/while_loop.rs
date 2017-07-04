#![feature(plugin)]
#![plugin(clippy)]

#![warn(while_let_loop, empty_loop, while_let_on_iterator)]
#![allow(dead_code, never_loop, unused, cyclomatic_complexity)]

fn main() {
    let y = Some(true);
    loop {
        if let Some(_x) = y {
            let _v = 1;
        } else {
            break
        }
    }
    loop { // no error, break is not in else clause
        if let Some(_x) = y {
            let _v = 1;
        }
        break;
    }
    loop {
        match y {
            Some(_x) => true,
            None => break
        };
    }
    loop {
        let x = match y {
            Some(x) => x,
            None => break
        };
        let _x = x;
        let _str = "foo";
    }
    loop {
        let x = match y {
            Some(x) => x,
            None => break,
        };
        { let _a = "bar"; };
        { let _b = "foobar"; }
    }
    loop { // no error, else branch does something other than break
        match y {
            Some(_x) => true,
            _ => {
                let _z = 1;
                break;
            }
        };
    }
    while let Some(x) = y { // no error, obviously
        println!("{}", x);
    }

    // #675, this used to have a wrong suggestion
    loop {
        let (e, l) = match "".split_whitespace().next() {
            Some(word) => (word.is_empty(), word.len()),
            None => break
        };

        let _ = (e, l);
    }

    let mut iter = 1..20;
    while let Option::Some(x) = iter.next() {
        println!("{}", x);
    }

    let mut iter = 1..20;
    while let Some(x) = iter.next() {
        println!("{}", x);
    }

    let mut iter = 1..20;
    while let Some(_) = iter.next() {}

    let mut iter = 1..20;
    while let None = iter.next() {} // this is fine (if nonsensical)

    let mut iter = 1..20;
    if let Some(x) = iter.next() { // also fine
        println!("{}", x)
    }

    // the following shouldn't warn because it can't be written with a for loop
    let mut iter = 1u32..20;
    while let Some(x) = iter.next() {
        println!("next: {:?}", iter.next())
    }

    // neither can this
    let mut iter = 1u32..20;
    while let Some(x) = iter.next() {
        println!("next: {:?}", iter.next());
    }

    // or this
    let mut iter = 1u32..20;
    while let Some(x) = iter.next() {break;}
    println!("Remaining iter {:?}", iter);

    // or this
    let mut iter = 1u32..20;
    while let Some(x) = iter.next() {
        iter = 1..20;
    }
}

// regression test (#360)
// this should not panic
// it's okay if further iterations of the lint
// cause this function to trigger it
fn no_panic<T>(slice: &[T]) {
    let mut iter = slice.iter();
    loop {
        let _ = match iter.next() {
            Some(ele) => ele,
            None => break
        };
        loop {}
    }
}

fn issue1017() {
    let r: Result<u32, u32> = Ok(42);
    let mut len = 1337;

    loop {
        match r {
            Err(_) => len = 0,
            Ok(length) => {
                len = length;
                break
            }
        }
    }
}

// Issue #1188
fn refutable() {
    let a = [42, 1337];
    let mut b = a.iter();

    // consume all the 42s
    while let Some(&42) = b.next() {
    }

    let a = [(1, 2, 3)];
    let mut b = a.iter();

    while let Some(&(1, 2, 3)) = b.next() {
    }

    let a = [Some(42)];
    let mut b = a.iter();

    while let Some(&None) = b.next() {
    }

    /* This gives “refutable pattern in `for` loop binding: `&_` not covered”
    for &42 in b {}
    for &(1, 2, 3) in b {}
    for &Option::None in b.next() {}
    // */

    let mut y = a.iter();
    loop { // x is reused, so don't lint here
        while let Some(v) = y.next() {
        }
    }

    let mut y = a.iter();
    for _ in 0..2 {
        while let Some(v) = y.next() {
        }
    }
}
