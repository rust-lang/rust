//@ known-bug: rust-lang/rust#146577
trait Trait {
    type Assoc;
}

fn foo(f: impl Fn(<() as Trait>::Assoc)) {
    (|x| f(x))()
}

fn main() {}
