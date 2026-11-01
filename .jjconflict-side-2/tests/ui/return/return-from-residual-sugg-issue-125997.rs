//@ run-rustfix

#![allow(unused_imports)]
#![allow(dead_code)]

use std::fs::File;
use std::io::prelude::*;

fn test1() {
    let mut _file = File::create("foo.txt")?;
    //~^ ERROR the `?` operator can only be used in a function
}

fn test2() {
    let mut _file = File::create("foo.txt")?;
    //~^ ERROR the `?` operator can only be used in a function
    println!();
}

macro_rules! mac {
    () => {
        fn test3() {
            let mut _file = File::create("foo.txt")?;
            //~^ ERROR the `?` operator can only be used in a function
            println!();
        }
    };
}

struct A;

impl A {
    fn test4(&self) {
        let mut _file = File::create("foo.txt")?;
        //~^ ERROR the `?` operator can only be used in a method
    }

    fn test5(&self) {
        let mut _file = File::create("foo.txt")?;
        //~^ ERROR the `?` operator can only be used in a method
        println!();
    }
}

fn main() {
    let mut _file = File::create("foo.txt")?;
    //~^ ERROR the `?` operator can only be used in a function
    mac!();
}
