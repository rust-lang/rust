fn foo(a: [0; 1]) {} //~ ERROR expected type, found `0`
//~| ERROR expected one of `)`, `,`, `->`, `where`, or `{`, found `]`
// FIXME(jseyfried): avoid emitting the second error (preexisting)

fn main() {}
