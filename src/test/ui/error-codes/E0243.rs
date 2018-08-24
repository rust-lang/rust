struct Foo<T> { x: T }
struct Bar { x: Foo }
                //~^ ERROR wrong number of type arguments: expected 1, found 0 [E0243]

fn main() {
}
