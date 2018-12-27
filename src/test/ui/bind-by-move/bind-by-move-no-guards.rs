use std::sync::mpsc::channel;

fn main() {
    let (tx, rx) = channel();
    let x = Some(rx);
    tx.send(false);
    match x {
        Some(z) if z.recv().unwrap() => { panic!() },
            //~^ ERROR cannot bind by-move into a pattern guard
        Some(z) => { assert!(!z.recv().unwrap()); },
        None => panic!()
    }
}
