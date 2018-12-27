#[repr(u128)]
enum Foo { //~ ERROR E0658
    Bar(u64),
}

fn main() {}
