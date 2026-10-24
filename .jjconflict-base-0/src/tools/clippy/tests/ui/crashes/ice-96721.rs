macro_rules! foo {
    () => {
        "bar.rs"
    };
}

#[path = foo!()] //~ ERROR: malformed `path` attribute
mod abc {}

fn main() {}
