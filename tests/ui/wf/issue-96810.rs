struct S<T: Tr>(T::Assoc);

trait Tr {
    type Assoc;
}

struct Hoge<K> {
    s: S<K>, //~ ERROR trait `Tr` is not implemented for `K`
    a: u32,
}

fn main() {}
