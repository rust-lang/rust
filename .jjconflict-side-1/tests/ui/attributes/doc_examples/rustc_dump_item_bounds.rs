//@ dont-require-annotations: ERROR
//@ compile-flags: --crate-type lib -Z ui-testing=no

#![feature(rustc_attrs)]

trait Trait<T> {
    #[rustc_dump_item_bounds]
    type Assoc: PartialEq<String>;
}
