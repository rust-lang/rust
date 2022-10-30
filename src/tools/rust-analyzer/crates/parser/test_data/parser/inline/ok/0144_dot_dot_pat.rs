fn main() {
    let .. = ();
    //
    // Tuples
    //
    let (a, ..) = ();
    let (a, ..,) = ();
    let Tuple(a, ..) = ();
    let Tuple(a, ..,) = ();
    let (.., ..) = ();
    let Tuple(.., ..) = ();
    let (.., a, ..) = ();
    let Tuple(.., a, ..) = ();
    //
    // Slices
    //
    let [..] = ();
    let [head, ..] = ();
    let [head, tail @ ..] = ();
    let [head, .., cons] = ();
    let [head, mid @ .., cons] = ();
    let [head, .., .., cons] = ();
    let [head, .., mid, tail @ ..] = ();
    let [head, .., mid, .., cons] = ();
}
