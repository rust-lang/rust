#![feature(plugin)]
#![plugin(clippy)]

#![deny(while_let_loop, empty_loop, while_let_on_iterator)]
#![allow(dead_code, unused)]

fn main() {
    let y = Some(true);
    loop { //~ERROR
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
    loop { //~ERROR
        match y {
            Some(_x) => true,
            None => break
        };
    }
    loop { //~ERROR
        let x = match y {
            Some(x) => x,
            None => break
        };
        let _x = x;
        let _str = "foo";
    }
    loop { //~ERROR
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

    let mut iter = 1..20;
    while let Option::Some(x) = iter.next() { //~ERROR this loop could be written as a `for` loop
        println!("{}", x);
    }

    let mut iter = 1..20;
    while let Some(x) = iter.next() { //~ERROR this loop could be written as a `for` loop
        println!("{}", x);
    }

    let mut iter = 1..20;
    while let Some(_) = iter.next() {} //~ERROR this loop could be written as a `for` loop

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
    loop { //~ERROR
        let _ = match iter.next() {
            Some(ele) => ele,
            None => break
        };
        loop {} //~ERROR empty `loop {}` detected.
    }
}
