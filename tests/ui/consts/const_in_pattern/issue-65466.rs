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
        C => (), //~ ERROR constant of non-structural type `&[O<B>]` in a pattern
        _ => (),
    }
}
