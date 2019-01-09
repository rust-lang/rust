pub struct List<T> {
    head: T,
}

impl Drop for List<i32> { //~ ERROR E0366
    fn drop(&mut self) {
        panic!()
    }
}

fn main() {
    List { head: 0 };
}
