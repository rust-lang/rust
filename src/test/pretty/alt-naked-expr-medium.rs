// pretty-exact

fn main() {
    let x = Some(3);
    let _y =
        match x {
          Some(_) => ~[~"some(_)", ~"not", ~"SO", ~"long", ~"string"],
          None => ~[~"none"]
        };
}
