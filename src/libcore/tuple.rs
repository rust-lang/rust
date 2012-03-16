#[doc = "Operations on tuples"];

#[doc = "Return the first element of a pair"]
pure fn first<T:copy, U:copy>(pair: (T, U)) -> T {
    let (t, _) = pair;
    ret t;
}

#[doc = "Return the second element of a pair"]
pure fn second<T:copy, U:copy>(pair: (T, U)) -> U {
    let (_, u) = pair;
    ret u;
}

#[doc = "Return the results of swapping the two elements of a pair"]
pure fn swap<T:copy, U:copy>(pair: (T, U)) -> (U, T) {
    let (t, u) = pair;
    ret (u, t);
}


#[test]
fn test_tuple() {
    assert first((948, 4039.48)) == 948;
    assert second((34.5, "foo")) == "foo";
    assert swap(('a', 2)) == (2, 'a');
}

