#[derive(Debug)]
struct Pair<T, V> (T, V);

impl Pair<
    &str,
    isize
> {
    fn say(self: &Pair<&str, isize>) {
//~^ ERROR mismatched `self` parameter type
//~| ERROR mismatched `self` parameter type
        println!("{:?}", self);
    }
}

fn main() {
    let result = &Pair("shane", 1);
    result.say();
}
