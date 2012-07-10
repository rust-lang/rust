// pretty-exact

fn main() {
    let x = some(3);
    let _y =
        alt x {
          some(_) => ~[~"some(_)", ~"not", ~"SO", ~"long", ~"string"],
          none => ~[~"none"]
        };
}
