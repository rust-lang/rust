#![expect(incomplete_features)]
#![feature(min_generic_const_args)]

fn takes_empty_array<const A: []>() {}
//~^ ERROR: expected type, found `]`

fn main() {
    takes_empty_array::<{ [] }>();
}
