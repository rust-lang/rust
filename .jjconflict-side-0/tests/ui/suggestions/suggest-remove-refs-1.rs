//@ run-rustfix

fn main() {
    let v = vec![0, 1, 2, 3];

    for (i, _) in &v.iter().enumerate() {
        //~^ ERROR `&Enumerate<std::slice::Iter<'_, {integer}>>` is not an iterator
        println!("{}", i);
    }
}
