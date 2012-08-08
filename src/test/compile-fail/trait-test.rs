trait foo { fn foo(); }

impl uint: int { fn foo() {} } //~ ERROR trait

fn main() {}
