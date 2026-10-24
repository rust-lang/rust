// Tests that anonymous parameters are a hard error in edition 2018.

//@ edition:2018

trait T {
    fn foo(i32); //~ ERROR expected one of `:`, `@`, or `|`, found `)`

    // Also checks with `&`
    fn foo_with_ref(&mut i32);
    //~^ ERROR expected one of `:`, `@`, or `|`, found `)`

    fn foo_with_qualified_path(<Bar as T>::Baz);
    //~^ ERROR expected one of `(`, `...`, `..=`, `..`, `::`, `:`, `{`, or `|`, found `)`

    fn foo_with_qualified_path_and_ref(&<Bar as T>::Baz);
    //~^ ERROR expected one of `(`, `...`, `..=`, `..`, `::`, `:`, `{`, or `|`, found `)`

    fn foo_with_multiple_qualified_paths(<Bar as T>::Baz, <Bar as T>::Baz);
    //~^ ERROR expected one of `(`, `...`, `..=`, `..`, `::`, `:`, `{`, or `|`, found `,`
    //~| ERROR expected one of `(`, `...`, `..=`, `..`, `::`, `:`, `{`, or `|`, found `)`

    fn bar_with_default_impl(String, String) {}
    //~^ ERROR expected one of `:`
    //~| ERROR expected one of `:`

    // do not complain about missing `b`
    fn baz(a:usize, b, c: usize) -> usize { //~ ERROR expected one of `:`
        a + b + c
    }
}

fn main() {}
