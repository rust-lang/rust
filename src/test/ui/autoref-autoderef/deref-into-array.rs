// check-pass

struct Test<T>([T; 1]);

impl<T> std::ops::Deref for Test<T> {
    type Target = [T; 1];

    fn deref(&self) -> &[T; 1] {
        &self.0
    }
}

fn main() {
    let out = Test([(); 1]);
    let blah = out.len();
    println!("{}", blah);
}
