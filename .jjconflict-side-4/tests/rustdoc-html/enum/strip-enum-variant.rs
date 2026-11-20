//@ has strip_enum_variant/enum.MyThing.html
//@ has - '//code' 'Shown'
//@ !has - '//code' 'NotShown'
//@ has - '//code' '// some variants omitted'
// Also check that `NotShown` isn't displayed in the sidebar.
//@ snapshot no-not-shown - '//*[@class="sidebar-elems"]/section/*[@class="block variant"]'
pub enum MyThing {
    Shown,
    #[doc(hidden)]
    NotShown,
}
