fn main() {
    #[inline] struct Foo;  //~ ERROR attribute should be applied to function or closure
    #[repr(C)] fn foo() {} //~ ERROR attribute should be applied to a struct, enum, or union
}
