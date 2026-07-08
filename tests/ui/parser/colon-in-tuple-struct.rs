// Suggest the user to use double colon when there's a colon in tuple struct

struct Foo(std::string:String);
//~^ ERROR expected one of `!`, `(`, `)`, `+`, `,`, `::`, or `<`, found `:`
//~| HELP if you meant to write a path, use a double colon
