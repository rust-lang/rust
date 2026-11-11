#![expect(incomplete_features)]
#![feature(gca_min_const_items)]

fn takes_empty_array<const A: []>() {}
//~^ ERROR: expected type, found `]`

fn main() {
    takes_empty_array::<{ [] }>();
}
