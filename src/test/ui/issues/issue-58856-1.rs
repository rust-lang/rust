impl A {
    //~^ ERROR cannot find type `A` in this scope
    fn b(self>
    //~^ ERROR expected one of `)`, `,`, or `:`, found `>`
    //~| ERROR expected `;` or `{`, found `>`
}

fn main() {}
