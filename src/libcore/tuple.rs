/*
Module: tuple
*/

// FIXME #1546: Would rather write fst<T, U>(+pair: (T, U)) -> T
fn first<T:copy, U:copy>(pair: (T, U)) -> T {
    let (t, _) = pair;
    ret t;
}

fn second<T:copy, U:copy>(pair: (T, U)) -> U {
    let (_, u) = pair;
    ret u;
}

fn swap<T:copy, U:copy>(pair: (T, U)) -> (U, T) {
    let (t, u) = pair;
    ret (u, t);
}


#[test]
fn test_tuple() {
    assert first((948, 4039.48)) == 948;
    assert second((34.5, "foo")) == "foo";
    assert swap(('a', 2)) == (2, 'a');
}

