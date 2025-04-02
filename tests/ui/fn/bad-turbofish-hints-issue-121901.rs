// Regression test for the parser wrongfully suggesting turbofish syntax in below syntax errors

type One = for<'a> fn(Box<dyn Send + 'a);
//~^ ERROR: expected one of `+`, `,`, or `>`, found `)`
type Two = for<'a> fn(Box<dyn Send + 'a>>);
//~^ ERROR: unmatched angle bracket

fn main() {}
