#[derive(Debug)]
struct Pair<T, V> (T, V);

impl Pair<
    &str,
    isize
> {
    fn say(self: &Pair<&str, isize>) {
//~^ ERROR mismatched method receiver
//~| ERROR mismatched method receiver
        println!("{:?}", self);
    }
}

fn main() {
    let result = &Pair("shane", 1);
    result.say();
}
