//@ aux-build:issue-81943-lib.rs
extern crate issue_81943_lib as lib;

fn f<F: Fn(i32)>(f: F) { f(0); }
fn g(t: i32) -> i32 { t }
fn main() {
  f(|x| lib::d!(x)); //~ERROR
  f(|x| match x { tmp => { g(tmp) } }); //~ERROR
  macro_rules! d {
    ($e:expr) => { match $e { x => { g(x) } } } //~ERROR
  }
  f(|x| d!(x));
}
