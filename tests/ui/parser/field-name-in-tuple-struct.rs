// Provide diagnostics when the user writes field names in tuple struct.(issue#144595)

struct Foo(a:u8,b:u8);
//~^ ERROR expected one of `!`, `(`, `)`, `+`, `,`, `::`, or `<`, found `:`
//~| HELP if you meant to write a path, use a double colon:
//~| HELP if you meant to create a regular struct, use curly braces:
