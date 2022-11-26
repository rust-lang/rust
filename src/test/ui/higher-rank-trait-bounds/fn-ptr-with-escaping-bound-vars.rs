// check-pass

fn main() {
    foo();
    foo2();
}

fn foo()
where
    for<'a> for<'b> fn(&'a (), &'b ()): Fn(&'a (), &'static ()),
{
}

fn foo2()
where
    for<'a> fn(&'a ()): Fn(&'a ()),
{
}
