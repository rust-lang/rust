//@ check-pass
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver


fn main() {
    let vec: Vec<Box<dyn Trait>> = Vec::new();

    for i in vec {
        i.fn_2();
    }
}

trait OtherTrait {}

trait Trait {
    fn fn_1(&self) -> impl OtherTrait
    where
        Self: Sized;

    fn fn_2(&self) -> bool;
}
