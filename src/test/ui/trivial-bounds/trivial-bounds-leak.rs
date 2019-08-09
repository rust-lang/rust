// Check that false bounds don't leak
#![feature(trivial_bounds)]

pub trait Foo {
    fn test(&self);
}

fn return_str() -> str where str: Sized {
    *"Sized".to_string().into_boxed_str()
}

fn cant_return_str() -> str { //~ ERROR
    *"Sized".to_string().into_boxed_str()
}

fn my_function() where i32: Foo
{
    3i32.test();
    Foo::test(&4i32);
    generic_function(5i32);
}

fn foo() {
    3i32.test(); //~ ERROR
    Foo::test(&4i32); //~ ERROR
    generic_function(5i32); //~ ERROR
}

fn generic_function<T: Foo>(t: T) {}

fn main() {}
