//~ ERROR overflow evaluating the requirement `Map<&mut std::ops::Range<u8>, {closure@$DIR/issue-83150.rs:14:24: 14:27}>: Iterator`
//@ build-fail
//@ compile-flags: -Copt-level=0
//@ normalize-stderr-test: "long-type-\d+" -> "long-type-hash"
//@ ignore-compare-mode-next-solver (hangs)

fn main() {
    let mut iter = 0u8..1;
    func(&mut iter)
}

fn func<T: Iterator<Item = u8>>(iter: &mut T) {
    //~^ WARN function cannot return without recursing
    func(&mut iter.map(|x| x + 1))
}
