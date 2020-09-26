#![deny(indirect_structural_match)]

// check-pass

#[derive(PartialEq, Eq)]
enum O<T> {
    Some(*const T), // Can also use PhantomData<T>
    None,
}

struct B;

const C: &[O<B>] = &[O::None];

fn main() {
    let x = O::None;
    match &[x][..] {
        C => (),
        _ => (),
    }
}
