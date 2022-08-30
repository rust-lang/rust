fn opaque<T>(t: T) -> impl Sized {
    //~^ ERROR cannot resolve opaque type
    //~| WARNING function cannot return without recursing
    opaque(Some(t))
}

#[allow(dead_code)]
fn main() {}
