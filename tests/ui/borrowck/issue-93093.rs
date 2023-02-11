// edition:2018
struct S {
    foo: usize,
}
impl S {
    async fn bar(&self) { //~ HELP consider changing this to be a mutable reference
        //~| SUGGESTION &mut self
        self.foo += 1; //~ ERROR cannot assign to `self.foo`, which is behind a `&` reference [E0594]
    }
}

fn main() {
    S { foo: 1 }.bar();
}
