impl A {
//~^ ERROR cannot find type `A` in this scope
    fn b(self>
    //~^ ERROR expected one of `)`, `,`, or `:`, found `>`
    //~| ERROR expected one of `->`, `where`, or `{`, found `>`
    //~| ERROR expected one of `->`, `async`, `const`, `crate`, `default`, `existential`,
}

fn main() {}
