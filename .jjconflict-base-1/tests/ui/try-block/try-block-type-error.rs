//@ edition: 2018

#![feature(try_blocks)]

fn foo() -> Option<()> { Some(()) }

fn main() {
    let _: Option<f32> = try {
        foo()?;
        42
        //~^ ERROR type mismatch
    };

    let _: Option<i32> = try {
        foo()?;
    };
    //~^ ERROR type mismatch
}
