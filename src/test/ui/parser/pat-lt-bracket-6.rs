fn main() {
    let Test(&desc[..]) = x; //~ ERROR: expected one of `)`, `,`, or `@`, found `[`
    //~^ ERROR cannot find value `x` in this scope
    //~| ERROR cannot find tuple struct/variant `Test` in this scope
    //~| ERROR subslice patterns are unstable
}
