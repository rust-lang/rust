// pretty-exact

// actually this doesn't quite look how I want it to, but I can't
// get the prettyprinter to indent the long expr

fn main() {
    let x = Some(3);
    let y =
        match x {
          Some(_) =>
          ~"some" + ~"very" + ~"very" + ~"very" + ~"very" + ~"very" +
              ~"very" + ~"very" + ~"very" + ~"long" + ~"string",
          None => ~"none"
        };
    assert y == ~"some(_)";
}
