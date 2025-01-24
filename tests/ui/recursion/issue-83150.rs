//~ ERROR overflow evaluating the requirement `Map<&mut std::ops::Range<u8>, {closure@$DIR/issue-83150.rs:13:24: 13:27}>: Iterator`
//@ build-fail
//@ compile-flags: -Copt-level=0
//@ normalize-stderr: "long-type-\d+" -> "long-type-hash"

fn main() {
    let mut iter = 0u8..1;
    func(&mut iter)
}

fn func<T: Iterator<Item = u8>>(iter: &mut T) {
    //~^ WARN function cannot return without recursing
    func(&mut iter.map(|x| x + 1))
}
