// revisions: stock if_match

#![cfg_attr(if_match, feature(const_if_match))]

fn main() {
    enum Foo {
        Drop = assert_eq!(1, 1)
        //[stock,if_match]~^ ERROR if may be missing an else clause
        //[stock]~^^ ERROR `match` is not allowed in a `const`
        //[stock]~| ERROR `match` is not allowed in a `const`
        //[stock]~| ERROR `if` is not allowed in a `const`
    }
}
