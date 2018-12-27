// run-pass
// Regression test for #31299. This was generating an overflow error
// because of eager normalization:
//
// proving `M: Sized` requires
// - proving `PtrBack<Vec<M>>: Sized` requires
//   - normalizing `Vec<<Vec<M> as Front>::Back>>: Sized` requires
//     - proving `Vec<M>: Front` requires
//       - `M: Sized` <-- cycle!
//
// If we skip the normalization step, though, everything goes fine.
//
// This could be fixed by implementing lazy normalization everywhere.
//
// However, we want this to work before then. For that, when checking
// whether a type is Sized we only check that the tails are Sized. As
// PtrBack does not have a tail, we don't need to normalize anything
// and this compiles

trait Front {
    type Back;
}

impl<T> Front for Vec<T> {
    type Back = Vec<T>;
}

struct PtrBack<T: Front>(Vec<T::Back>);

struct M(PtrBack<Vec<M>>);

fn main() {
    std::mem::size_of::<M>();
}
