fn foo() {
    let x = Tr<A, A:>;
    //~^ ERROR expected one of `!`, `.`, `::`, `;`, `?`, `else`, `{`, or an operator, found `,`
}

fn main() {}
