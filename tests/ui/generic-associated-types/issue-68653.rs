// A regression test for #68653, which was fixed by #68938.

//@ check-pass

trait Fun {
    type F<'a: 'a>;
}

impl <T> Fun for T {
    type F<'a> = Self;
}

fn main() {}
