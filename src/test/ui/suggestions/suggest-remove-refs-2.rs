fn main() {
    let v = vec![0, 1, 2, 3];

    for (i, n) in & & & & &v.iter().enumerate() {
        //~^ ERROR `&&&&&std::iter::Enumerate<std::slice::Iter<'_, {integer}>>` is not an iterator
        println!("{}", i);
    }
}
