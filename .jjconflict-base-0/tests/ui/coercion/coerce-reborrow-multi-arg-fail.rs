fn test<T>(_a: T, _b: T) {}

fn main() {
    test(&mut 7, &7);
    //~^ ERROR mismatched types
}
