/// docs for foo
#[deprecated(since = "1.2.3", note = "text")]
#[macro_export]
macro_rules! foo {
    ($($tt:tt)*) => {}
}

//@ has macro_by_example/macros/index.html
pub mod macros {
    //@ !hasraw - 'pub use foo as bar;'
    //@ has macro_by_example/macros/macro.bar.html
    //@ has - '//*[@class="docblock"]' 'docs for foo'
    //@ has - '//*[@class="stab deprecated"]' 'Deprecated since 1.2.3: text'
    //@ has - '//a/@href' 'macro_by_example.rs.html#4-6'
    #[doc(inline)]
    pub use foo as bar;
}
