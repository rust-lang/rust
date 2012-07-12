// error-pattern:expected `str/~` but found `int`

const i: str = 10i;
fn main() { log(debug, i); }
