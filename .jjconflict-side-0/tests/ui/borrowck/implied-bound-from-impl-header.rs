//@ check-pass
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

// Make sure that we can normalize `<T as Ref<'a>>::Assoc` to `&'a T` and get
// its implied bounds in impl header.

trait Ref<'a> {
    type Assoc;
}
impl<'a, T> Ref<'a> for T where T: 'a {
    type Assoc = &'a T;
}

fn outlives<'a, T: 'a>() {}

trait Trait<'a, T> {
    fn test();
}

impl<'a, T> Trait<'a, T> for <T as Ref<'a>>::Assoc {
    fn test() {
        outlives::<'a, T>();
    }
}

fn main() {}
