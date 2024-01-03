// run-rustfix
#![deny(bare_trait_objects)]
fn ord_prefer_dot(s: String) -> Ord {
    //~^ ERROR the trait `Ord` cannot be made into an object
    (s.starts_with("."), s)
}
fn main() {
    let _ = ord_prefer_dot(String::new());
}
