use std::ops::Index;

struct MyType;
impl Index<usize> for MyType {
    type Output = String;
    fn index(&self, _idx: usize) -> &String {
        const { &String::new() }
    }
}

fn main() {
    let x = MyType;
    let y = &x[0];
    y.push_str("");
    //~^ ERROR cannot borrow `*y` as mutable, as it is behind a `&` reference
}
