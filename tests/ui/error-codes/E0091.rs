type Foo<T> = u32; //~ ERROR E0091
type Foo2<A, B> = Box<A>; //~ ERROR E0091

fn main() {
}
