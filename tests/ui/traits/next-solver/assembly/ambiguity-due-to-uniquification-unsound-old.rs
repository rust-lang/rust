// A regression test for trait-system-refactor-initiative#307 affecting
// the old solver instead. The old solver only cares about region identity
// inside of `project` when merging candidates. This means this unsoundness
// also affects the old solver.

//@ revisions: current next
//@[next] compile-flags: -Znext-solver
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[current] failure-status: 101
//@[current] dont-check-compiler-stderr
//@[current] known-bug: unknown
trait Sup<'a, T> {
    type Assoc;
    fn to_assoc(&self) -> Self::Assoc;
}
impl<'a> Sup<'a, ()> for &'a str {
    type Assoc = &'a str;
    fn to_assoc(&self) -> Self::Assoc { self }
}

// Need the trait arguments to pass WF-checking
trait Tr<'a, 'b, T, U>: Sup<'a, T, Assoc = &'a str> + Sup<'b, U, Assoc = &'b str> {}
impl<'a> Tr<'a, 'a, (), ()> for &'a str {}

trait HideMe {
    fn hide_me(&self) -> &'static str;
}

// Need the ambiguous `T: Sup<'a, (), Assoc = &'static str>` bound to only be
// used inside of the trait impl, as otherwise we also encounter ambiguity
// during normalization, which ICEs instead of ignoring it.
impl<'a, T: Sup<'a, (), Assoc = &'static str> + ?Sized> HideMe for T {
    fn hide_me(&self) -> &'static str {
        self.to_assoc()
    }
}
fn yeet<'a>(x: &dyn Tr<'a, 'a, (), ()>) -> &'static str {
    x.hide_me()
    //[next]~^ ERROR: type annotations needed: cannot satisfy `dyn Tr<'a, 'a, (), ()>: Sup<'a, ()>`
}

fn main() {
    let s = String::from("hello");
    let r = yeet(&s.as_str());
    drop(s);
    println!("{r}");
}
