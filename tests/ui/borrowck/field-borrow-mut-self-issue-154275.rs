struct S {
    a: i32,
}

impl S {
    fn helper(&mut self) {}

    fn f(&mut self) {
        let a = &self.a;
        self.helper(); //~ ERROR cannot borrow `*self` as mutable because it is also borrowed as immutable
        a;
    }
}

fn main() {}
