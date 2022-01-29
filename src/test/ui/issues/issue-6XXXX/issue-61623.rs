fn f1<'a>(_: &'a mut ()) {}

fn f2<P>(_: P, _: ()) {}

fn f3<'a>(x: &'a ((), &'a mut ())) {
    f2(|| x.0, f1(x.1))
//~^ ERROR cannot borrow `*x.1` as mutable, as it is behind a `&` reference
//~| ERROR cannot borrow `*x.1` as mutable because it is also borrowed as immutable
}

fn main() {}
