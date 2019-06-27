// compile-flags: -Z chalk

trait Foo { }

struct S<T: Foo> {
    x: T,
}

impl Foo for i32 { }
impl<T> Foo for Option<T> { }

fn main() {
    let s = S {
       x: 5,
    };

    let s = S { //~ ERROR the trait bound `{float}: Foo` is not satisfied
        x: 5.0,
    };

    let s = S {
        x: Some(5.0),
    };
}
