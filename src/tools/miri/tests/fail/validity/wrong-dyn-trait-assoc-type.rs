trait Trait {
    type Assoc;
    fn foo(&self) -> Self::Assoc;
}

impl<T: Copy> Trait for T {
    type Assoc = T;
    fn foo(&self) -> T {
        *self
    }
}

fn main() {
    let v: Box<dyn Trait<Assoc = u8>> = Box::new(2);
    let v: Box<dyn Trait<Assoc = bool>> = unsafe { std::mem::transmute(v) }; //~ERROR: wrong trait

    if v.foo() {
        println!("huh");
    }
}
