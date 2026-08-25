//@ aux-build:close_window.rs
//@ revisions: ascii unicode
//@[unicode] compile-flags: -Zunstable-options --error-format=human-unicode

extern crate close_window;

fn main() {
   let s = close_window::S;
    s.method();
   //[ascii]~^ ERROR method `method` is private
}
