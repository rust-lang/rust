// check-pass

#![feature(const_trait_impl)]
#![feature(const_fn_trait_bound)]
#![feature(const_precise_live_drops)]

const fn foo<T, E>(res: Result<T, E>) -> Option<T> where E: ~const Drop {
    match res {
        Ok(t) => Some(t),
        Err(_e) => None,
    }
}

pub struct Foo<T>(T);

const fn baz<T: ~const Drop, E: ~const Drop>(res: Result<Foo<T>, Foo<E>>) -> Option<Foo<T>> {
    foo(res)
}

fn main() {}
