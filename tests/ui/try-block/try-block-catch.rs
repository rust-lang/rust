//@ edition: 2018

#![feature(try_blocks)]

fn main() {
    let res: Option<bool> = try {
        true
    } catch { };
    //~^ ERROR keyword `catch` cannot follow a `try` block
}
