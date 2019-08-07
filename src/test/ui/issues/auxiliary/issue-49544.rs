#![crate_type = "lib"]

pub fn foo() -> Vec<String> {
    std::env::args()
        .skip(1)
        .collect()
}
