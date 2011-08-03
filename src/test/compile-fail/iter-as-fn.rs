// error-pattern:calling iter outside of for each loop
iter i() { }
fn main() { i(); }
