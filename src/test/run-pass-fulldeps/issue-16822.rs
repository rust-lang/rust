// aux-build:issue-16822.rs

extern crate issue_16822 as lib;

use std::cell::RefCell;

struct App {
    i: isize
}

impl lib::Update for App {
    fn update(&mut self) {
        self.i += 1;
    }
}

fn main(){
    let app = App { i: 5 };
    let window = lib::Window { data: RefCell::new(app) };
    window.update(1);
}
