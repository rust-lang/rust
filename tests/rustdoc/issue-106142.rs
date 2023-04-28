// @has 'issue_106142/a/index.html'
// @count 'issue_106142/a/index.html' '//ul[@class="item-table"]//li//a' 1

#![allow(rustdoc::broken_intra_doc_links)]

pub mod a {
    /// [`m`]
    pub fn f() {}

    #[macro_export]
    macro_rules! m {
        () => {};
    }
}
