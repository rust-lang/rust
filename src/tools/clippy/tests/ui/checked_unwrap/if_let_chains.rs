//@require-annotations-for-level: ERROR
#![deny(clippy::unnecessary_unwrap)]

#[clippy::msrv = "1.85"]
fn if_let_chains_unsupported(a: Option<u32>, b: Option<u32>) {
    if a.is_none() || b.is_none() {
        println!("a or b is not set");
    } else {
        println!("the value of a is {}", a.unwrap());
        //~^ unnecessary_unwrap
        //~| HELP: try using `match`
    }
}

#[clippy::msrv = "1.88"]
fn if_let_chains_supported(a: Option<u32>, b: Option<u32>) {
    if a.is_none() || b.is_none() {
        println!("a or b is not set");
    } else {
        println!("the value of a is {}", a.unwrap());
        //~^ unnecessary_unwrap
        //~| HELP: try using `if let` or `match`
    }
}
