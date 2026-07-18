pub struct Dep1;
pub struct Dep2;
pub struct Dep3;
pub struct Dep4;

//@ hasraw crates.js 'dep1'
//@ hasraw search.index/name/*.js 'Dep1'
//@ has dep1/index.html
#[doc(alias = "dep1_missing")]
pub struct Dep5;
