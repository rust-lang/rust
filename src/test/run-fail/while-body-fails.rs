// error-pattern:quux
fn main() { let x: int = { while true { fail ~"quux"; } ; 8 } ; }
