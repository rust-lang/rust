#![feature(plugin)]

#![plugin(clippy)]
#![deny(clippy)]
#![allow(unused)]

fn single_match(){
    let x = Some(1u8);
    match x {  //~ ERROR you seem to be trying to use match
               //~^ HELP try
        Some(y) => {
            println!("{:?}", y);
        }
        _ => ()
    }
    // Not linted
    match x {
        Some(y) => println!("{:?}", y),
        None => ()
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
        false => { println!("Noooo!"); },
    };
    
    match test { //~ ERROR you seem to be trying to match on a boolean expression
        false => { println!("Noooo!"); },
        _ => (),
    };
    
    match test { //~ ERROR you seem to be trying to match on a boolean expression
        false => { println!("Noooo!"); },
        true => { println!("Yes!"); },
    };

    // Not linted
    match option {
        1 ... 10 => (),
        10 ... 20 => (),
        _ => (),
    };
}

fn ref_pats() {
    {
        let v = &Some(0);
        match v {  //~ERROR instead of prefixing all patterns with `&`
            &Some(v) => println!("{:?}", v),
            &None => println!("none"),
        }
        match v {  // this doesn't trigger, we have a different pattern
            &Some(v) => println!("some"),
            other => println!("other"),
        }
    }
    let tup =& (1, 2);
    match tup {  //~ERROR instead of prefixing all patterns with `&`
        &(v, 1) => println!("{}", v),
        _ => println!("none"),
    }
    // special case: using & both in expr and pats
    let w = Some(0);
    match &w {  //~ERROR you don't need to add `&` to both
        &Some(v) => println!("{:?}", v),
        &None => println!("none"),
    }
    // false positive: only wildcard pattern
    let w = Some(0);
    match w {
        _ => println!("none"),
    }
}

fn main() {
}
