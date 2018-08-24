// compile-flags: -Z parse-only

impl Foo; //~ ERROR expected one of `!`, `(`, `+`, `::`, `<`, `for`, `where`, or `{`, found `;`
