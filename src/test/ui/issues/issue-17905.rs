#[derive(Debug)]
struct Pair<T, V> (T, V);

impl Pair<
    &str, //~ ERROR missing lifetime specifier
    isize
> {
    fn say(self: &Pair<&str, isize>) {
        println!("{}", self);
    }
}

fn main() {
    let result = &Pair("shane", 1);
    result.say();
}
