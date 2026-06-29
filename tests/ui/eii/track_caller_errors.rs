// `#[track_caller]` is derived onto EII implementations during codegen, so:
//   - implementations must not repeat it manually;
//   - it is only valid on functions, not statics.
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

#[track_caller] //~ ERROR `#[track_caller]` attribute cannot be used on foreign statics
#[eii(sfoo)]
static FOO: u64 = 42;

fn main() {
    decl1(4);
    println!("{FOO}");
}
