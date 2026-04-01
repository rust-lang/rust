#![feature(return_type_notation)]

fn function() {}

fn not_a_method()
where
    function(..): Send,
    //~^ ERROR expected function, found function `function`
    //~| ERROR return type notation not allowed in this position yet
{
}

fn not_a_method_and_typoed()
where
    function(): Send,
    //~^ ERROR expected type, found function `function`
{
}

trait Tr {
    fn method();
}

// Forgot the `T::`
fn maybe_method_overlaps<T: Tr>()
where
    method(..): Send,
    //~^ ERROR cannot find function `method` in this scope
    //~| ERROR return type notation not allowed in this position yet
{
}

// Forgot the `T::`, AND typoed `(..)` to `()`
fn maybe_method_overlaps_and_typoed<T: Tr>()
where
    method(): Send,
    //~^ ERROR cannot find type `method` in this scope
{
}

fn main() {}
