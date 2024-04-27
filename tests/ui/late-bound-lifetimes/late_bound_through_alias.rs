//@ check-pass

fn f(_: X) -> X {
    unimplemented!()
}

fn g<'a>(_: X<'a>) -> X<'a> {
    unimplemented!()
}

type X<'a> = &'a ();

fn main() {
    let _: for<'a> fn(X<'a>) -> X<'a> = g;
    let _: for<'a> fn(X<'a>) -> X<'a> = f;
}
