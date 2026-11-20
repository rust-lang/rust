//@ compile-flags: -Zassumptions-on-binders -Znext-solver=globally

fn foo<'a>(_a: &'a u32)
where
    for<'b> &'b (): 'a,
{
}

fn main() {
    foo(&10);
    //~^ ERROR: higher-ranked lifetime bound could not be satisfied
}
