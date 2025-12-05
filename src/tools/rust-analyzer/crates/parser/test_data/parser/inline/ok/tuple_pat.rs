fn main() {
    let (a, b, ..) = ();
    let (a,) = ();
    let (..) = ();
    let () = ();
    let (| a | a, | b) = ((),());
}
