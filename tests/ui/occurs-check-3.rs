// From Issue #778

enum Clam<T> { A(T) }
fn main() {
    let c;
    c = Clam::A(c);
    //~^ ERROR overflow setting `Clam<_>` to a subtype of `_`
    match c {
        Clam::A::<isize>(_) => { }
    }
}
