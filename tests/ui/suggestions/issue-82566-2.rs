struct Foo1<const N1: usize>;
struct Foo2<const N1: usize, const N2: usize>;
struct Foo3<const N1: usize, const N2: usize, const N3: usize>;

impl<const N1: usize> Foo1<N1> {
    const SUM: usize = N1;
}

impl<const N1: usize, const N2: usize> Foo2<N1, N2> {
    const SUM: usize = N1 + N2;
}

impl<const N1: usize, const N2: usize, const N3: usize> Foo3<N1, N2, N3> {
    const SUM: usize = N1 + N2 + N3;
}

fn foo1() -> [(); Foo1<10>::SUM] { //~ ERROR: comparison operators cannot be chained
    todo!()
}

fn foo2() -> [(); Foo2<10, 20>::SUM] {
    //~^ ERROR: expected one of `.`, `?`, `]`, or an operator, found `,`
    todo!()
}

fn foo3() -> [(); Foo3<10, 20, 30>::SUM] {
    //~^ ERROR: expected one of `.`, `?`, `]`, or an operator, found `,`
    todo!()
}

fn main() {}
