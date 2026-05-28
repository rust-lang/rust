// Regression test for #58451:
//
// Error reporting here encountered an ICE in the shift to universes.

fn f<I>(i: I)
where
    I: IntoIterator,
    I::Item: for<'a> Into<&'a ()>,
{}

fn main() {
    f(&[f()]); //~ ERROR function takes 1 argument
}
