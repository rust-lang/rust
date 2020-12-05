// run-pass

#[derive(Debug)]
struct Pair<T, V> (T, V);

impl Pair<
    &str,
    isize
> {
    fn say(&self) {
        println!("{:?}", self);
    }
}

fn main() {
    let result = &Pair("shane", 1);
    result.say();
}
