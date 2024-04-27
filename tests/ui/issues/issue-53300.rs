// issue 53300

pub trait A {
    fn add(&self, b: i32) -> i32;
}

fn addition() -> Wrapper<impl A> {}
//~^ ERROR cannot find type `Wrapper` in this scope [E0412]

fn main() {
    let res = addition();
}
