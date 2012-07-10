// pretty-exact

// actually this doesn't quite look how I want it to, but I can't
// get the prettyprinter to indent the long expr

fn main() {
    let x = some(3);
    let y =
        alt x {
          some(_) =>
          "some" + "very" + "very" + "very" + "very" + "very" + "very" +
              "very" + "very" + "long" + "string",

          none => "none"
        };
    assert y == "some(_)";
}
