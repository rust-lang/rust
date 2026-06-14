// There is a regression introduced for issue #143828
//@ edition: 2015

trait Foo {
    fn foo([a, b]: [i32; 2]) {}
    //~^ ERROR expected `;` or `]`, found `,`
    //~| ERROR parameters can't have complex patterns in associated functions in traits in Rust 2015
}

fn main() {}
