//@ known-bug: #106473
#![feature(generic_const_exprs)]

const DEFAULT: u32 = 1;

struct V<const U: usize = DEFAULT>
where
    [(); U]:;

trait Tr {}

impl Tr for V {}
