#[forbid(non_camel_case_types)]
enum foo { //~ ERROR type, variant, or trait must be camel case
    Bar
}

fn main() {
}
