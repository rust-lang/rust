//@ build-fail
//@ compile-flags: -Zwrite-long-types-to-disk=yes

#![allow(dead_code)]

enum Foo<A> {
    Fst,
    Snd(Box<dyn Fn() -> Foo<(A, A)>>),
}

fn recursive<A>(x: Foo<A>) {
    match x {
        Foo::Fst => (),
        Foo::Snd(f) => recursive(f()),
        //~^ ERROR reached the recursion limit while instantiating
    }
}

fn main() {
    let p0: Foo<()> = Foo::Fst;
    recursive(p0);
}
