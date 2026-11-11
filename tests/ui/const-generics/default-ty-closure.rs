// https://github.com/rust-lang/rust/issues/116796

struct X<const FN: fn() = { || {} }>;
//~^ ERROR using function pointers as const generic parameters is forbidden

fn main() {}
