#![feature(share_trait)]

use std::clone::Share;

struct NotClone;

impl Share for NotClone {}
//~^ ERROR the trait bound `NotClone: Clone` is not satisfied

fn main() {}
