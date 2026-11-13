// Regression test for #154826.

use std::sync::LazyLock;

static V: LazyLock<(Vec<u8>,)> = LazyLock::new(|| (vec![],));

fn main() {
    let (v,) = *V;
    //~^ ERROR cannot move out of dereference of `LazyLock<(Vec<u8>,)>`
    //~| HELP consider borrowing here
    let _: &Vec<_> = &v;
}
