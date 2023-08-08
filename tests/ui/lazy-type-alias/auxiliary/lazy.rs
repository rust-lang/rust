#![feature(lazy_type_alias)]
#![allow(incomplete_features)]

pub type Alias<T: Copy> = Option<T>;
