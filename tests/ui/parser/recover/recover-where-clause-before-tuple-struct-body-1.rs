// Regression test for issues #100790 and #106439.

// Make sure that we still show a helpful error message even if the trailing semicolon is missing.

struct Foo<T> where T: MyTrait, (T)
//~^ ERROR where clauses are not allowed before tuple struct bodies
//~| ERROR expected `;`, found `<eof>`
