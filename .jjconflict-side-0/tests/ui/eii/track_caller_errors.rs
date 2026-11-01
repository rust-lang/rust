// `#[track_caller]` is derived onto EII implementations during codegen, so
// implementations must not repeat it manually.
#![feature(extern_item_impls)]

#[eii]
fn decl1(x: u64) {
    println!("default {x}");
}

#[track_caller] //~ ERROR `#[decl1]` is not allowed to have `#[track_caller]`
#[decl1]
fn impl1(x: u64) {
    println!("explicit {x}");
}

fn main() {
    decl1(4);
}
