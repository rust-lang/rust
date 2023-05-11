// @has issue_108925/enum.MyThing.html
// @has - '//code' 'Shown'
// @!has - '//code' 'NotShown'
// @!has - '//code' '// some variants omitted'
#[non_exhaustive]
pub enum MyThing {
    Shown,
    #[doc(hidden)]
    NotShown,
}
