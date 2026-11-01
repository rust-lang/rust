struct S;
impl S {
    fn foo() {}
    fn bar(&self) {
        self.foo(); //~ ERROR no method named `foo` found for reference `&S` in the current scope
        let f: fn() = self.foo; //~ ERROR no field `foo` on type `&S`
    }
}

fn main() {}
