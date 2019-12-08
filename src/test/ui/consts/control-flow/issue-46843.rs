// revisions: stock if_match

#![cfg_attr(if_match, feature(const_if_match))]

enum Thing { This, That }

fn non_const() -> Thing {
    Thing::This
}

pub const Q: i32 = match non_const() {
    //[stock]~^ ERROR `match` is not allowed in a `const`
    //[if_match]~^^ ERROR calls in constants are limited to constant functions
    Thing::This => 1,
    Thing::That => 0
};

fn main() {}
