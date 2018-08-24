struct Foo { x: bool }
struct Bar<S, T> { x: Foo<S, T> }
                      //~^ ERROR wrong number of type arguments: expected 0, found 2 [E0244]


fn main() {
}
