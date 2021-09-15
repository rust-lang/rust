// check-pass
#![feature(const_trait_impl)]
#![feature(const_fn_trait_bound)]

struct S;

trait A {}
trait B {}

impl const A for S {}
impl const B for S {}

impl S {
    const fn a<T: ~const A>() where T: ~const B {

    }
}

const _: () = S::a::<S>();

fn main() {}
