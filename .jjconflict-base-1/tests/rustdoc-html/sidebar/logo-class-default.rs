// Note: this test is paired with logo-class.rs and logo-class-rust.rs.
//@ !has logo_class_default/struct.SomeStruct.html '//*[@class="logo-container"]/img' ''
//@ !has src/logo_class_default/logo-class-default.rs.html '//*[@class="sub-logo-container"]/img' ''
pub struct SomeStruct;
