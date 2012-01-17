
fn fst<T, U>(+pair: (T, U)) -> T {
    let (t, _) = pair;
    ret t;
}

fn snd<T, U>(+pair: (T, U)) -> U {
    let (_, u) = pair;
    ret u;
}
