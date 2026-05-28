//@ check-pass

fn lifetime<'a>()
where
    &'a (): 'a,
{
    /* do nothing */
}

fn doesnt_work()
where
    for<'a> &'a (): 'a,
{
    /* do nothing */
}

fn main() {
    lifetime();
    doesnt_work();
}
