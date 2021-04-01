#![crate_name = "foo"]

pub struct MyStruct {
    pub my_field: i32,
}

impl MyStruct {
    /// [`MyStruct::my_field()`] gets a reference to [`field@MyStruct::my_field`].
    /// What about [without disambiguators](MyStruct::my_field)?
    // @has foo/struct.MyStruct.html '//a[@href="../foo/struct.MyStruct.html#method.my_field"]' 'MyStruct::my_field()'
    // @has - '//a[@href="../foo/struct.MyStruct.html#structfield.my_field"]' 'MyStruct::my_field'
    // @!has - '//a' 'field'
    // @has - '//a[@href="../foo/struct.MyStruct.html#method.my_field"]' 'without disambiguators'
    pub fn my_field(&self) -> i32 { self.bar }
}

pub enum MyEnum {
    MyVariant { my_field: i32 },
}

impl MyEnum {
    /// [`MyEnum::my_field()`] gets a reference to [`field@MyEnum::MyVariant::my_field`].
    /// What about [without disambiguators](MyEnum::MyVariant::my_field)?
    // @has foo/enum.MyEnum.html '//a[@href="../foo/enum.MyEnum.html#method.my_field"]' 'MyEnum::MyVariant::my_field()'
    // @has - '//a[@href="../foo/enum.MyEnum.html#enumfield.my_field"]' 'MyEnum::MyVariant::my_field'
    // @!has - '//a' 'field'
    // @has - '//a[@href="../foo/enum.MyEnum.html#method.my_field"]' 'without disambiguators'
    pub fn bar(&self) -> i32 { self.bar }
}
