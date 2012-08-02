//! Operations on tuples

trait tuple_ops<T,U> {
    pure fn first() -> T;
    pure fn second() -> U;
    pure fn swap() -> (U, T);
}

impl extensions <T:copy, U:copy> of tuple_ops<T,U> for (T, U) {

    /// Return the first element of self
    pure fn first() -> T {
        let (t, _) = self;
        return t;
    }

    /// Return the second element of self
    pure fn second() -> U {
        let (_, u) = self;
        return u;
    }

    /// Return the results of swapping the two elements of self
    pure fn swap() -> (U, T) {
        let (t, u) = self;
        return (u, t);
    }

}

trait extended_tuple_ops<A,B> {
    fn zip() -> ~[(A, B)];
    fn map<C>(f: fn(A, B) -> C) -> ~[C];
}

impl extensions<A: copy, B: copy> of extended_tuple_ops<A,B>
        for (&[A], &[B]) {

    fn zip() -> ~[(A, B)] {
        let (a, b) = self;
        vec::zip(a, b)
    }

    fn map<C>(f: fn(A, B) -> C) -> ~[C] {
        let (a, b) = self;
        vec::map2(a, b, f)
    }
}

impl extensions<A: copy, B: copy> of extended_tuple_ops<A,B>
        for (~[A], ~[B]) {

    fn zip() -> ~[(A, B)] {
        let (a, b) = self;
        vec::zip(a, b)
    }

    fn map<C>(f: fn(A, B) -> C) -> ~[C] {
        let (a, b) = self;
        vec::map2(a, b, f)
    }
}

#[test]
fn test_tuple() {
    assert (948, 4039.48).first() == 948;
    assert (34.5, ~"foo").second() == ~"foo";
    assert ('a', 2).swap() == (2, 'a');
}

