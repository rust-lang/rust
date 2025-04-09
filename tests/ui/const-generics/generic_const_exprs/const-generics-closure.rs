// compile-fail
// Regression test for issue #127424

fn bar() -> impl Into<
    [u8; {
        f_t(&*s); //~ ERROR cannot find function `f_t` in this scope

        c(&*s); //~ ERROR cannot find function `c` in this scope

        c(&*s); //~ ERROR cannot find function `c` in this scope

        struct X;

        c1(*x); //~ ERROR cannot find function `c1` in this scope
        let _ = for<'a, 'b> |x: &'a &'a Vec<&'b u32>, b: bool| -> &'a Vec<&'b u32> { *x }; //~ ERROR `for<...>` binders for closures are experimental
    }],
> {
    [99]
}
