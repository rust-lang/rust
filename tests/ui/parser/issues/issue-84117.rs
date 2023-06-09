fn main() {
    let outer_local:e_outer<&str, { let inner_local:e_inner<&str, }
    //~^ ERROR expected one of `>`, a const expression
    //~| ERROR expected one of `>`, a const expression, lifetime, or type, found `}`
    //~| ERROR expected one of `!`, `.`, `::`, `;`, `?`, `else`, `{`, or an operator, found `,`
    //~| ERROR expected one of `!`, `.`, `::`, `;`, `?`, `else`, `{`, or an operator, found `,`
    //~| ERROR expected one of `!`, `.`, `::`, `;`, `?`, `else`, `{`, or an operator, found `,`
}
//~^ ERROR expected one of `,` or `>`, found `}`
