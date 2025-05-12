fn main() {
    let fields = vec![1];
    let variant = vec![2];

    // should not suggest `*&variant.iter()`
    for (src, dest) in std::iter::zip(fields.iter(), &variant.iter()) {
        //~^ ERROR `&std::slice::Iter<'_, {integer}>` is not an iterator
        //~| ERROR `&std::slice::Iter<'_, {integer}>` is not an iterator
        eprintln!("{} {}", src, dest);
    }

    // don't suggest add `variant.iter().clone().clone()`
    for (src, dest) in std::iter::zip(fields.iter(), &variant.iter().clone()) {
        //~^ ERROR `&std::slice::Iter<'_, {integer}>` is not an iterator
        //~| ERROR `&std::slice::Iter<'_, {integer}>` is not an iterator
        eprintln!("{} {}", src, dest);
    }
}
