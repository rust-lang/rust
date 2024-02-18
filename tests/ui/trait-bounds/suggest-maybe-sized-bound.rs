// issue: 120878
fn main() {
    struct StructA<A, B = A> {
        _marker: std::marker::PhantomData<fn() -> (A, B)>,
    }

    struct StructB {
        a: StructA<isize, [u8]>,
        //~^ ERROR: the size for values of type `[u8]` cannot be known at compilation time [E0277]
    }

    trait Trait {
        type P<X>;
    }

    impl Trait for () {
        type P<X> = [u8];
        //~^ ERROR: the size for values of type `[u8]` cannot be known at compilation time [E0277]
    }
}
