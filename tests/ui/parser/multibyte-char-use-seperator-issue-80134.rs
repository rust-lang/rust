// Regression test for #80134.

fn main() {
    (()é);
    //~^ ERROR: expected one of `)`, `,`, `.`, `?`, or an operator
    //~| ERROR: cannot find value `é` in this scope
    (()氷);
    //~^ ERROR: expected one of `)`, `,`, `.`, `?`, or an operator
    //~| ERROR: cannot find value `氷` in this scope
}
