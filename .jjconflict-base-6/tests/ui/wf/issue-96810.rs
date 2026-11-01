struct S<T: Tr>(T::Assoc);

trait Tr {
    type Assoc;
}

struct Hoge<K> {
    s: S<K>, //~ ERROR the trait bound `K: Tr` is not satisfied
    a: u32,
}

fn main() {}
