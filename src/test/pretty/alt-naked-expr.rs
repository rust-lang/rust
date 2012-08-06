// pretty-exact

fn main() {
    let x = some(3);
    let y = match x { some(_) => ~"some(_)", none => ~"none" };
    assert y == ~"some(_)";
}
