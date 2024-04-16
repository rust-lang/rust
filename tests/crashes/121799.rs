//@ known-bug: #121799
struct S {
    _: str
}

fn func(a: S)
{
    let _x = a.f;
}

fn main() {}
