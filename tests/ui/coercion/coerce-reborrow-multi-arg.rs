//@ build-pass
fn test<T>(_a: T, _b: T) {}

fn main() {
    test(&7, &7);
    test(&7, &mut 7);
    test::<&i32>(&mut 7, &7);
    test::<&i32>(&mut 7, &mut 7);
}
