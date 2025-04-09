// Regression test for issue #127424

fn bar() -> impl Into<
    [u8; {
        //~^ ERROR mismatched types [E0308]
        f_t(&*s);
        //~^ ERROR cannot find function `f_t` in this scope [E0425]
        //~| ERROR cannot find value `s` in this scope [E0425]

        c(&*s);
        //~^ ERROR cannot find function `c` in this scope [E0425]
        //~| ERROR cannot find value `s` in this scope [E0425]

        c(&*s);
        //~^ ERROR cannot find function `c` in this scope [E0425]
        //~| ERROR cannot find value `s` in this scope [E0425]

        struct X;

        c1(*x);
        //~^ ERROR cannot find function `c1` in this scope [E0425]
        //~| ERROR cannot find value `x` in this scope [E0425]

        let _ = for<'a, 'b> |x: &'a &'a Vec<&'b u32>, b: bool| -> &'a Vec<&'b u32> { *x };
        //~^ ERROR `for<...>` binders for closures are experimental [E0658]
    }],
> {
    [99]
}
//~^ ERROR `main` function not found in crate `const_generics_closure` [E0601]
