// From Issue #778

enum Clam<T> { A(T) }
fn main() {
    let c;
    c = Clam::A(c);
    //~^ ERROR overflow assigning `Clam<_>` to `_`
    match c {
        Clam::A::<isize>(_) => { }
    }
}
