fn main() {
    #[inline] struct Foo;  //~ ERROR attribute cannot be used on
    #[repr(C)] fn foo() {} //~ ERROR attribute should be applied to a struct, enum, or union
}
