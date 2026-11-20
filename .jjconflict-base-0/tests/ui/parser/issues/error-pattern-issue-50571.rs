// There is a regression introduced for issue #143828
//@ edition: 2015

#![allow(dead_code)]
trait Foo {
    fn foo([a, b]: [i32; 2]) {}
    //~^ ERROR: expected `;` or `]`, found `,`
    //~| ERROR: patterns aren't allowed in methods without bodies
}

fn main() {}
