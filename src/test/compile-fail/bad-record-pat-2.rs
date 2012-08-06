// error-pattern:did not expect a record with a field `q`

fn main() { match {x: 1, y: 2} { {x: x, q: q} => { } } }
