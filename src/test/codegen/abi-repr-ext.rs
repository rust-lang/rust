#![crate_type="lib"]

#[repr(i8)]
pub enum Type {
    Type1 = 0,
    Type2 = 1
}

// CHECK: define{{( dso_local)?}} signext i8 @test()
#[no_mangle]
pub extern "C" fn test() -> Type {
    Type::Type1
}
