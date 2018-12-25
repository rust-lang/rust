// compile-flags: -Z parse-only

struct X<'a, T, 'b> {
//~^ ERROR lifetime parameters must be declared prior to type parameters
    x: &'a T
}
