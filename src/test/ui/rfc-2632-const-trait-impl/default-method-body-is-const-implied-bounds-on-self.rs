// check-pass

#![feature(const_trait_impl)]
#![feature(const_fn_trait_bound)]

pub trait Foo {
    #[default_method_body_is_const]
    fn do_stuff(self) where Self: Sized {
        do_stuff_as_foo(self);
    }
}

const fn do_stuff_as_foo<T: ~const Foo>(foo: T) {
    std::mem::forget(foo);
}

fn main() {}