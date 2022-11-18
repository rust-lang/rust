// build-fail
// compile-flags: -Copt-level=0

fn main() {
    let mut iter = 0u8..1;
    func(&mut iter)
}

fn func<T: Iterator<Item = u8>>(iter: &mut T) { //~ WARN function cannot return without recursing
    func(&mut iter.map(|x| x + 1))
    //~^ ERROR reached the recursion limit while instantiating `func::<Map<&mut Map<&mut Map<&mu...n/issue-83150.rs:11:24: 11:27]>>`
}
