fn foo(a: [0; 1]) {} //~ ERROR expected type, found `0`
//~| ERROR expected one of `)`, `,`, `->`, `;`, `where`, or `{`, found `]`

fn main() {}
