struct Bar {}

fn main() {
    let old: Option<Bar> = None;

    let b = Bar {
        ..old.as_ref().unwrap() //~ ERROR E0308
    };
}
