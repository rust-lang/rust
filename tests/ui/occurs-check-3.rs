// From Issue #778

enum Clam<T> { A(T) }
fn main() {
    let c;
    c = Clam::A(c);
    //~^ ERROR overflow evaluating the requirement `Clam<_> <: _`
    match c {
        Clam::A::<isize>(_) => { }
    }
}
