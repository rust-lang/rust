#![crate_name = "bar"]

pub struct Ref<'a>(&'a u32);

pub fn test5(a: &u32) -> Ref {
    Ref(a)
}

pub fn test6(a: &u32) -> Ref<'_> {
    Ref(a)
}
