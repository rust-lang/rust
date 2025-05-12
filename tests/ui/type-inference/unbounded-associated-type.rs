trait T {
    type A;
    fn foo(&self) -> Self::A {
        panic!()
    }
}

struct S<X>(std::marker::PhantomData<X>);

impl<X> T for S<X> {
   type A = X;
}

fn main() {
    S(std::marker::PhantomData).foo(); //~ ERROR type annotations needed
}
