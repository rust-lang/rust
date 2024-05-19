// Regression test for #121532
// Checks the we don't ICE in ArrayIntoIter
// lint when typeck has failed

 // Typeck fails for the arg type as
 // `Self` makes no sense here
fn func(a: Self::ItemsIterator) { //~ ERROR failed to resolve: `Self` is only available in impls, traits, and type definitions
    a.into_iter();
}

fn main() {}
