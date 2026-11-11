//@ check-pass

#![crate_type = "lib"]
#![deny(invalid_doc_attributes)]
#![doc(test(no_crate_inject))]
#![doc(test(attr(deny(warnings))))]
#![doc(test())]

mod test {
    #![doc(test(attr(allow(warnings))))]
}

#[doc(test(attr(allow(dead_code))))]
static S: u32 = 5;

#[doc(test(attr(allow(dead_code))))]
const C: u32 = 5;

#[doc(test(attr(deny(dead_code))))]
struct A {
    #[doc(test(attr(allow(dead_code))))]
    field: u32
}

#[doc(test(attr(deny(dead_code))))]
union U {
    #[doc(test(attr(allow(dead_code))))]
    field: u32,
    field2: u64,
}

#[doc(test(attr(deny(dead_code))))]
enum Enum {
    #[doc(test(attr(allow(dead_code))))]
    Variant1,
}

#[doc(test(attr(deny(dead_code))))]
impl A {
    #[doc(test(attr(deny(dead_code))))]
    fn method() {}
}

#[doc(test(attr(deny(dead_code))))]
trait MyTrait {
    #[doc(test(attr(deny(dead_code))))]
    fn my_trait_fn();
}

#[doc(test(attr(deny(dead_code))))]
pub fn foo() {}
