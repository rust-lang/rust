//@ revisions: unchecked-by-default checked-by-default
//@ check-pass

#![feature(marked_unchecked_type_aliases)]
#![cfg_attr(checked_by_default, feature(checked_type_aliases))]

#[rustc_unchecked_type_alias]
type Alias0<T: Copy> = T; //~ WARN [type_alias_bounds]

#[rustc_unchecked_type_alias]
type Alias1 = [(); panic!()]; // OK

fn main() {
    let _: Alias0<String>; // OK
}
