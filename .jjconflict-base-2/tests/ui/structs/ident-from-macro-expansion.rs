struct Foo {
    inner: Inner,
}

struct Inner {
    y: i32,
}

macro_rules! access {
    ($expr:expr, $ident:ident) => {
        $expr.$ident
    }
}

fn main() {
    let k = Foo { inner: Inner { y: 0 } };
    access!(k, y);
    //~^ ERROR no field `y` on type `Foo`
}
