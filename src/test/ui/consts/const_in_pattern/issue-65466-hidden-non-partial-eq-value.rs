#![deny(indirect_structural_match)]

// Sometimes the code-generation for pattern-matching a `const` will generate a
// invocation of `PartialEq::eq` (rather than unfolding the structure of that
// constant inline, which is the usual compilation strategy used in rustc
// currently...)
//
// The difference between unfolding inline and invoking `PartialEq::eq` is not
// meant to be observable, because of [RFC 1445][] (c.f. #31434): "Restrict
// constants in patterns"
//
// [RFC 1445]: https://github.com/rust-lang/rfcs/pull/1445
//
// However, there are cases where rustc's static analysis of the pattern itself
// was not thorough enough, and it would let patterns through that do end up
// invoking `PartialEq::eq` on constants that *do not even implement*
// `PartialEq` in the first place.
//
// We had been relying on the structural-match checking to catch such cases,
// where we would first run the structural match check, and if we found an ADT
// that was a non-structural-match, then we would *then* check if the `const` in
// question even implements `PartialEq`.
//
// But that strategy, of a delayed check for `PartialEq`, does not suffice in
// every case. Here is an example.

#[derive(PartialEq, Eq)]
enum O<T> {
    Some(*const T), // Can also use PhantomData<T>
    None,
}

struct B;

const C: &[O<B>] = &[O::None];

pub fn foo() {
    let x = O::None;
    match &[x][..] {
        C => (),
        //~^ ERROR `B` must be annotated with `#[derive(PartialEq, Eq)]`
        _ => (),
    }
}

fn main() { }
