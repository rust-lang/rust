#[derive(Debug)]
struct Foo { }

struct S {f:String}
impl Drop for S {
    fn drop(&mut self) { println!("{}", self.f); }
}

fn main() {
    match (S {f:"foo".to_string()}) {
        S {f:_s} => {} //~ ERROR cannot move out
    }
}
