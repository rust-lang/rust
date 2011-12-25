// error-pattern:wrong type in main function: found 'fn(&&{x: int,y: int})'
fn main(foo: {x: int, y: int}) { }
