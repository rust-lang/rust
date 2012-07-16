//! Operations on tuples


impl extensions <T:copy, U:copy> for (T, U) {

    /// Return the first element of self
    pure fn first() -> T {
        let (t, _) = self;
        ret t;
    }

    /// Return the second element of self
    pure fn second() -> U {
        let (_, u) = self;
        ret u;
    }

    /// Return the results of swapping the two elements of self
    pure fn swap() -> (U, T) {
        let (t, u) = self;
        ret (u, t);
    }

}


#[test]
fn test_tuple() {
    assert (948, 4039.48).first() == 948;
    assert (34.5, ~"foo").second() == ~"foo";
    assert ('a', 2).swap() == (2, 'a');
}

