// Note: this test is paired with logo-class.rs.
// @has logo_class_default/struct.SomeStruct.html '//*[@class="logo-container"]/img[@class="rust-logo"]' ''
// @has src/logo_class_default/logo-class-default.rs.html '//*[@class="sub-logo-container"]/img[@class="rust-logo"]' ''
pub struct SomeStruct;
