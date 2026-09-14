#![feature(share_trait)]

use std::clone::Share;

fn require_share<T: Share>() {}

fn main() {
    require_share::<&mut i32>();
    //~^ ERROR the trait bound `&mut i32: Share` is not satisfied

    require_share::<String>();
    //~^ ERROR the trait bound `String: Share` is not satisfied

    require_share::<Vec<i32>>();
    //~^ ERROR the trait bound `Vec<i32>: Share` is not satisfied

    require_share::<Box<i32>>();
    //~^ ERROR the trait bound `Box<i32>: Share` is not satisfied
}
