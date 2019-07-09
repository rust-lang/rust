fn main() {
    for thing(x[]) in foo {} //~ ERROR: expected one of `)`, `,`, or `@`, found `[`
    //~^ ERROR cannot find value `foo` in this scope
    //~| ERROR cannot find tuple struct/variant `thing` in this scope
}
