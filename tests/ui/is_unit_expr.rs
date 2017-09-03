
#![feature(plugin)]
#[plugin(clippy)]
#[warn(unit_expr)]
#[allow(unused_variables)]
#[allow(no_effect)]

fn main() {
    let x = {
        "foo";
        "baz";
    };

}
