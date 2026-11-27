struct Qux(i32);

fn bad() {
    let mut map = std::collections::HashMap::new();
    map.insert(('a', 'b'), ('c', 'd'));

    for ((_, _), (&mut c, _)) in &mut map {
    //~^ ERROR mismatched types
        if c == 'e' {}
    }
}

fn bad2() {
    for Some(Qux(_)) | None in [Some(""), None] {
    //~^ ERROR mismatched types
        todo!();
    }
}

fn main() {}
