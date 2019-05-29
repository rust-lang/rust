// run-pass
/*

#5008 cast to &Trait causes code to segfault on method call

It fixes itself if the &Trait is changed to @Trait.
*/

trait Debuggable {
    fn debug_name(&self) -> String;
}

#[derive(Clone)]
struct Thing {
    name: String,
}

impl Thing {
    fn new() -> Thing { Thing { name: "dummy".to_string() } }
}

impl Debuggable for Thing {
    fn debug_name(&self) -> String { self.name.clone() }
}

fn print_name(x: &dyn Debuggable)
{
    println!("debug_name = {}", x.debug_name());
}

pub fn main() {
    let thing = Thing::new();
    print_name(&thing as &dyn Debuggable);
}
