#[derive(Default)]
struct Inner {
    a: u8,
    b: u8,
}

#[derive(Default)]
struct Outer {
    inner: Inner,
    defaulted: u8,
}

fn main(){
    Outer {
        //~^ ERROR missing field `defaulted` in initializer of `Outer`
        inner: Inner {
            a: 1,
            b: 2,
        }
        ..Default::default()
    };
}
