// aux-build:macro-in-other-crate.rs

#[macro_use] extern crate macro_in_other_crate;

macro_rules! local_mac {
    ($ident:ident) => { let $ident = 42; }
}

fn main() {
    let x = 2.0.neg();
    //~^ ERROR can't call method `neg` on ambiguous numeric type `{float}`

    let y = 2.0;
    let x = y.neg();
    //~^ ERROR can't call method `neg` on ambiguous numeric type `{float}`
    println!("{:?}", x);

    for i in 0..100 {
        println!("{}", i.pow(2));
        //~^ ERROR can't call method `pow` on ambiguous numeric type `{integer}`
    }

    local_mac!(local_bar);
    local_bar.pow(2);
    //~^ ERROR can't call method `pow` on ambiguous numeric type `{integer}`
}

fn qux() {
    mac!(bar);
    bar.pow(2);
    //~^ ERROR can't call method `pow` on ambiguous numeric type `{integer}`
}
