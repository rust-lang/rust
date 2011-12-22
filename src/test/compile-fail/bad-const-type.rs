// error-pattern:expected str but found int

const i: str = 10;
fn main() { log_full(core::debug, i); }
