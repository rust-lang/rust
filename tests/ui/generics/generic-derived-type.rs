//@ run-pass
fn g<X>(x: X) -> X { return x; }

#[derive(Clone)]
struct Pair<T> {
    a: T,
    b: T
}

fn f<T:Clone>(t: T) -> Pair<T> {
    let x: Pair<T> = Pair {a: t.clone(), b: t};
    return g::<Pair<T>>(x);
}

pub fn main() {
    let b = f::<isize>(10);
    println!("{}" ,b.a);
    println!("{}", b.b);
    assert_eq!(b.a, 10);
    assert_eq!(b.b, 10);
}
