fn foo(items: &mut Vec<u8>) {
    items.sort();
}

fn bar() -> impl Iterator<Item = i32> {
    //~^ ERROR expected `foo` to be a fn item that returns `i32`, but it returns `()` [E0271]
    let mut x: Vec<Vec<u8>> = vec![vec![0, 2, 1], vec![5, 4, 3]];
    x.iter_mut().map(foo)
}

fn main() {
    bar();
}
