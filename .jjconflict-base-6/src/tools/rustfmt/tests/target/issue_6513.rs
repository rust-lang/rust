#![feature(fn_delegation)]

struct Ty;
impl Ty {
    reuse std::convert::identity;
}

trait Trait {
    reuse std::convert::identity;
}
