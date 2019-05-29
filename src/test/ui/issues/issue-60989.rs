struct A {}
struct B {}

impl From<A> for B {
    fn from(a: A) -> B {
        B{}
    }
}

fn main() {
    let c1 = ();
    c1::<()>;
    //~^ ERROR type arguments are not allowed for this type

    let c1 = A {};
    c1::<dyn Into<B>>;
    //~^ ERROR type arguments are not allowed for this type
}
