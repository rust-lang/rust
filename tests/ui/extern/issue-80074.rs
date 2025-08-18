//@ edition:2018
//@ aux-crate:issue_80074=issue-80074-macro.rs
//@ aux-crate:issue_80074_2=issue-80074-macro-2.rs

#[macro_use]
extern crate issue_80074;

#[macro_use(m)]
extern crate issue_80074_2;
//~^^ ERROR: imported macro not found

fn main() {
    foo!();
    //~^ ERROR: macro `foo` is private
    //~| WARN: it will become a hard error in a future release!
    bar!();
    //~^ ERROR: cannot find macro `bar` in this scope
    m!();
    //~^ ERROR: cannot find macro `m` in this scope
}
