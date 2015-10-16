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


    while let Option::Some(x) = (1..20).next() { //~ERROR this loop could be written as a `for` loop
        println!("{}", x);
    }

    while let Some(x) = (1..20).next() { //~ERROR this loop could be written as a `for` loop
        println!("{}", x);
    }

    while let Some(_) = (1..20).next() {} //~ERROR this loop could be written as a `for` loop

    while let None = (1..20).next() {} // this is fine (if nonsensical)

    if let Some(x) = (1..20).next() { // also fine
        println!("{}", x)
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
