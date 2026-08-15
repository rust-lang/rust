// We used to not lower the extra `b @ ..` into `b @ _` which meant that no type
// was registered for the binding `b` although it passed through resolve.
// This resulted in an ICE (#69103).

fn main() {
    let [a @ .., b @ ..] = &mut [1, 2];
    //~^ ERROR `..` can only be used once per slice pattern
    b;

    let [.., c @ ..] = [1, 2];
    //~^ ERROR `..` can only be used once per slice pattern
    c;

    // This never ICEd, but let's make sure it won't regress either.
    let (.., d @ ..) = (1, 2);
    //~^ ERROR `..` patterns are not allowed here
    d;
}
