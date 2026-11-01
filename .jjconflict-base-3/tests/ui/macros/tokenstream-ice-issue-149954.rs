// Regression test for ICE https://github.com/rust-lang/rust/issues/149954
//@ edition: 2024

enum A {
    A
    const A: A = { //~ ERROR expected one of `(`, `,`, `=`, `{`, or `}`, found keyword `const`
        #[derive(Debug)]
        struct A
        where
            A: A<{ struct A<A: A<{ #[cfg] () }>> ; enum A  }
            //~^ ERROR malformed `cfg` attribute input
            //~| ERROR malformed `cfg` attribute input
            //~| ERROR expected trait, found struct `A`
            //~| ERROR expected trait, found type parameter `A`
            //~| ERROR expected trait, found struct `A`
            //~| ERROR expected trait, found type parameter `A`
            //~| ERROR expected one of `<`, `where`, or `{`, found `}`
            //~| ERROR expected one of `<`, `where`, or `{`, found `}`
            //~| ERROR expected one of `,`, `>`, or `}`, found `<eof>`
        }
        >;
}; //~ ERROR `main` function not found in crate
