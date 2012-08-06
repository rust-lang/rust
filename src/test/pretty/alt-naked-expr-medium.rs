// pretty-exact

fn main() {
    let x = some(3);
    let _y =
        match x {
          some(_) => ~[~"some(_)", ~"not", ~"SO", ~"long", ~"string"],
          none => ~[~"none"]
        };
}
