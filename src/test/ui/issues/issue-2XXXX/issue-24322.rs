struct B;

impl B {
    fn func(&self) -> u32 { 42 }
}

fn main() {
    let x: &fn(&B) -> u32 = &B::func; //~ ERROR mismatched types
}
