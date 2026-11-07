//@ edition: 2024

// Associated type bounds can be used within an `impl Trait`. These can be used
// as implied bounds of that associated type, but not of the opaque type itself.

#![forbid(unsafe_code)]

trait Foo {
    type Out;
    fn first(&self) -> u8;
}

impl<'x> Foo for &'x [u8] {
    type Out = ();
    fn first(&self) -> u8 {
        self[0]
    }
}

fn make<'x>(r: &'x [u8]) -> impl Foo<Out: 'static> {
    r
}

fn main() {
    let escaped;
    {
        let v: Vec<u8> = vec![42];
        escaped = make(&v);
        //~^ ERROR `v` does not live long enough
    }
    // Reuse the freed allocation: without the error above this prints `escaped = 7`.
    let reused: Vec<u8> = vec![7];
    println!("escaped = {}, reused = {reused:?}", escaped.first());
}
