// compile-flags: -Z parse-only

type closure = Box<lt/fn()>;
//~^ ERROR expected one of `!`, `(`, `+`, `,`, `::`, `<`, or `>`, found `/`
