#[allow(non_camel_case_types)]
struct foo;
struct Thing {
    foo: String,
}

fn example(t: Thing) {
    let Thing { foo } = t; //~ ERROR mismatched types
}

fn main() {}
