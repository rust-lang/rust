#![feature(error_type_id)]

#[derive(Debug)]
struct T;

impl std::fmt::Display for T {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "T")
    }
}

impl std::error::Error for T {
    fn type_id(&self) -> std::any::TypeId {
    //~^ ERROR cannot override `type_id` because it already has a `final` definition in the trait
        std::any::TypeId::of::<Self>()
    }
}

fn main() {}
