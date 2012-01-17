// FIXME #1546: Would rather write fst<T, U>(+pair: (T, U)) -> T
fn fst<T:copy, U:copy>(pair: (T, U)) -> T {
    let (t, _) = pair;
    ret t;
}

fn snd<T:copy, U:copy>(pair: (T, U)) -> U {
    let (_, u) = pair;
    ret u;
}
