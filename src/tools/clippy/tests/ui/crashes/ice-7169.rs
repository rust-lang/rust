#[derive(Default)]
struct A<T> {
    a: Vec<A<T>>,
    b: T,
}

fn main() {
    if let Ok(_) = Ok::<_, ()>(A::<String>::default()) {}
}
