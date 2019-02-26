#![feature(optin_builtin_traits)]

// @has issue_55321/struct.A.html
// @has - '//*[@id="implementations-list"]/*[@class="impl"]//code' "impl !Send for A"
// @has - '//*[@id="implementations-list"]/*[@class="impl"]//code' "impl !Sync for A"
pub struct A();

impl !Send for A {}
impl !Sync for A {}

// @has issue_55321/struct.B.html
// @has - '//*[@id="synthetic-implementations-list"]/*[@class="impl"]//code' "impl<T> !Send for \
// B<T>"
// @has - '//*[@id="synthetic-implementations-list"]/*[@class="impl"]//code' "impl<T> !Sync for \
// B<T>"
pub struct B<T: ?Sized>(A, Box<T>);
