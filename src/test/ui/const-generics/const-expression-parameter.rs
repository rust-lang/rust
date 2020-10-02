// revisions: full min

#![cfg_attr(full, feature(const_generics))]
#![cfg_attr(full, allow(incomplete_features))]
#![cfg_attr(min, feature(min_const_generics))]

fn i32_identity<const X: i32>() -> i32 {
    5
}

fn foo_a() {
    i32_identity::<-1>(); // ok
}

fn foo_b() {
    i32_identity::<1 + 2>(); //~ ERROR expected one of `,` or `>`, found `+`
}

fn foo_c() {
    i32_identity::< -1 >(); // ok
}

fn main() {
    i32_identity::<5>(); // ok
}
