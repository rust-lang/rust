// pp-exact

trait Tr { }
impl Tr for int;

fn foo(x: ~Tr: Freeze) -> ~Tr: Freeze { x }

fn main() {
    let x: ~Tr: Freeze;

    ~1 as ~Tr: Freeze;
}

