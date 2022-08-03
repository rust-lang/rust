// Regression test for issue 50714, make sure that this isn't a linker error.

fn main()
where
//~^ ERROR `main` function is not allowed to have a `where` clause
    fn(&()): Eq,
{
}
