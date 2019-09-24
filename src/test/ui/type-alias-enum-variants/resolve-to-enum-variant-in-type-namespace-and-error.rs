// Check that the compiler will resolve `<E>::V` to the variant `V` in the type namespace
// but will reject this because `enum` variants do not exist in the type namespace.

enum E {
    V
}

fn check() -> <E>::V {}
//~^ ERROR expected type, found variant `V`

fn main() {}
