// pretty-exact

fn main() {
    let x = Some(3);
    let y = match x { Some(_) => ~"some(_)", None => ~"none" };
    assert y == ~"some(_)";
}
