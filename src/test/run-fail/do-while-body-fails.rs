// error-pattern:quux
fn main() { let x: int = do  { fail "quux"; } while true; }
