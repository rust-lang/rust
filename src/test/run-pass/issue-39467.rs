macro_rules! expr { () => { () } }

enum A {}

impl A {
    const A: () = expr!();
}

fn main() {}
