// @has strip_enum_variant/enum.MyThing.html
// @has - '//code' 'Shown'
// @!has - '//code' 'NotShown'
// @has - '//code' '// some variants omitted'
pub enum MyThing {
    Shown,
    #[doc(hidden)]
    NotShown,
}
