// Regression test for issue #68550.
//
// The `&'static A:` where clause was triggering
// ICEs because it wound up being compiled to reference
// the `'empty(U0)` region.

fn run<'a, A>(x: A)
where
    A: 'static,
    &'static A: ,
{
    let _: &'a A = &x; //~ ERROR `x` does not live long enough
}

fn main() {}
