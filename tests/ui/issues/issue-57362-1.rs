// Test for issue #57362, ensuring that the self ty is shown in cases of higher-ranked lifetimes
// conflicts: the `expected` and `found` trait refs would otherwise be printed the same, leading
// to confusing notes such as:
//  = note: expected type `Trait`
//             found type `Trait`

// from issue #57362
trait Trait {
    fn f(self);
}

impl<T> Trait for fn(&T) {
    fn f(self) {
        println!("f");
    }
}

fn f() {
    let a: fn(_) = |_: &u8| {};
    a.f(); //~ ERROR no method named `f`
}

fn main() {}
