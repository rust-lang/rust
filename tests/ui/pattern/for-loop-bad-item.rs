struct Qux(i32);

fn good() {
    let mut map = std::collections::HashMap::new();
    map.insert(('a', 'b'), ('c', 'd'));

    for ((_, _), (&mut c, _)) in &mut map {
        if c == 'e' {}
    }
}

fn bad() {
    for Some(Qux(_)) | None in [Some(""), None] {
    //~^ ERROR mismatched types
        todo!();
    }
}

fn main() {}
