fn main() {
    let (c,p) = pipes::stream();
    let x = Some(p);
    c.send(false);
    match move x {
        Some(move z) if z.recv() => { fail }, //~ ERROR cannot bind by-move into a pattern guard
        Some(move z) => { assert !z.recv(); },
        None => fail
    }
}
