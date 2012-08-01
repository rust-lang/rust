// Issue #1761

impl of foo for int { fn foo() -> int { 10 } }
trait foo { fn foo() -> int; }
fn main() {}