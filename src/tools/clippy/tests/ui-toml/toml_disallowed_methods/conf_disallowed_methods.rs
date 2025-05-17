#![allow(clippy::needless_raw_strings)]
#![warn(clippy::disallowed_methods)]
#![allow(clippy::useless_vec)]

extern crate futures;
extern crate regex;

use futures::stream::{empty, select_all};
use regex::Regex;

use std::convert::identity;
use std::hint::black_box as renamed;

fn local_fn() {}

struct Struct;

impl Struct {
    fn method(&self) {}
}

trait Trait {
    fn provided_method(&self) {}
    fn implemented_method(&self);
}

impl Trait for Struct {
    fn implemented_method(&self) {}
}

mod local_mod {
    pub fn f() {}
}

fn main() {
    let re = Regex::new(r"ab.*c").unwrap();
    //~^ disallowed_methods
    re.is_match("abc");
    //~^ disallowed_methods

    let mut a = vec![1, 2, 3, 4];
    a.iter().sum::<i32>();
    //~^ disallowed_methods

    a.sort_unstable();
    //~^ disallowed_methods

    // FIXME(f16_f128): add a clamp test once the function is available
    let _ = 2.0f32.clamp(3.0f32, 4.0f32);
    //~^ disallowed_methods
    let _ = 2.0f64.clamp(3.0f64, 4.0f64);

    let indirect: fn(&str) -> Result<Regex, regex::Error> = Regex::new;
    //~^ disallowed_methods
    let re = indirect(".").unwrap();

    let in_call = Box::new(f32::clamp);
    //~^ disallowed_methods
    let in_method_call = ["^", "$"].into_iter().map(Regex::new);
    //~^ disallowed_methods

    // resolve ambiguity between `futures::stream::select_all` the module and the function
    let same_name_as_module = select_all(vec![empty::<()>()]);
    //~^ disallowed_methods

    local_fn();
    //~^ disallowed_methods
    local_mod::f();
    //~^ disallowed_methods
    let s = Struct;
    s.method();
    //~^ disallowed_methods
    s.provided_method();
    //~^ disallowed_methods
    s.implemented_method();
    //~^ disallowed_methods

    identity(());
    //~^ disallowed_methods
    renamed(1);
    //~^ disallowed_methods
}
