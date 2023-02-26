#![deny(map_unit_fn)]

fn foo(items: &mut Vec<u8>) {
    items.sort();
}

fn main() {
    let mut x: Vec<Vec<u8>> = vec![vec![0, 2, 1], vec![5, 4, 3]];
    x.iter_mut().map(foo);
    //~^ ERROR `Iterator::map` call that discard the iterator's values
    x.iter_mut().map(|items| {
    //~^ ERROR `Iterator::map` call that discard the iterator's values
        items.sort();
    });
    let f = |items: &mut Vec<u8>| {
        items.sort();
    };
    x.iter_mut().map(f);
    //~^ ERROR `Iterator::map` call that discard the iterator's values
}
