//@ edition: 2024

#[fused] //~ ERROR: the `#[fused]` attribute is an experimental feature
async fn foo() -> &'static str {
    "hello"
}

fn main() {}
