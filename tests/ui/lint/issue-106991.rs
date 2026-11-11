//@revisions: current next
//@[next] compile-flags: -Znext-solver=globally

fn foo(items: &mut Vec<u8>) {
    items.sort();
}

fn bar() -> impl Iterator<Item = i32> {
    //[current]~^ ERROR: expected `foo` to return `i32`, but it returns `()` [E0271]
    //[next]~^^ ERROR: expected `Map<IterMut<'_, Vec<u8>>, fn(&mut Vec<u8>) {foo}>` to be an iterator that yields `i32`, but it yields `()`
    let mut x: Vec<Vec<u8>> = vec![vec![0, 2, 1], vec![5, 4, 3]];
    x.iter_mut().map(foo)
}

fn baz() -> impl Iterator<Item = i32> {
    //[current]~^ ERROR: expected `IterMut<'_, Vec<u8>>` to be an iterator that yields `i32`, but it yields `&mut Vec<u8>`
    //[next]~^^ ERROR: expected `IterMut<'_, Vec<u8>>` to be an iterator that yields `i32`, but it yields `&mut Vec<u8>`
    let mut x: Vec<Vec<u8>> = vec![vec![0, 2, 1], vec![5, 4, 3]];
    if true {
        return x.iter_mut();
    }
    x.iter_mut()
}

fn main() {
    bar();
}
