// ignore-tidy-linelength

#![crate_name = "foo"]

#![feature(staged_api)]

#![stable(since="1.1.1", feature="rust1")]

#[stable(since="1.1.1", feature="rust1")]
pub struct SomeStruct;

impl SomeStruct {
    // @has 'foo/struct.SomeStruct.html' '//*[@id="SOME_CONST.v"]//div[@class="since"]' '1.1.2'
    #[stable(since="1.1.2", feature="rust2")]
    pub const SOME_CONST: usize = 0;
}
