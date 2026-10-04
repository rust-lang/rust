//@ proc-macro: derive-attribute-annotation-needed.rs
//@ edition: 2018


use derive_attribute_annotation_needed::Serialize;

fn f<T>() {}

#[derive(Serialize)]
pub struct Matrix {
    #[serde(serialize_with = "f")]
    //~^ ERROR type annotations needed
    matrix: (),
}

fn main() {
    f();
    //~^ ERROR type annotations needed
    //~| HELP consider specifying a concrete type for the type parameter `T`
}
