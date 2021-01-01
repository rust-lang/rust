struct Foo<B> {
    buffer: B
}

fn bar() {
    let Foo<Vec<u8>> //~ ERROR expected `;`, found `}`
    //~^ ERROR expected unit struct, unit variant or constant, found struct `Foo`
}

fn main() {}
