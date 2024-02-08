#![crate_name = "foo"]

//@ has 'foo/index.html'
//@ has - '//div[@id="rustdoc-toc"]/section/h3' 'Crate Items'

//@ has 'foo/bar/index.html'
//@ has - '//div[@id="rustdoc-toc"]/section/h3' 'Module Items'
pub mod bar {
    //@ has 'foo/bar/struct.Baz.html'
    //@ !has - '//div[@id="rustdoc-toc"]/section/h3' 'Module Items'
    pub struct Baz;
}

//@ has 'foo/baz/index.html'
//@ !has - '//div[@id="rustdoc-toc"]/section/h3' 'Module Items'
pub mod baz {}
