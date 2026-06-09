// Regression test for issue #127424

fn bar() -> impl Into<
    [u8; {
        //~^ ERROR mismatched types [E0308]
        let _ = for<'a, 'b> |x: &'a &'a Vec<&'b u32>, b: bool| -> &'a Vec<&'b u32> { *x };
        //~^ ERROR `for<...>` binders for closures are experimental [E0658]
    }],
> {
    [89]
}

fn main() {}
