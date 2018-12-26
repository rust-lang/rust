#![crate_name = "foo"]

// @has foo/struct.Foo0.html
pub struct Foo0;

impl Foo0 {
    // @has - '//*[@class="stab deprecated"]' 'Deprecated since 1.0.1: fn_with_doc'
    // @has - 'fn_with_doc short'
    // @has - 'fn_with_doc full'
    /// fn_with_doc short
    ///
    /// fn_with_doc full
    #[deprecated(since = "1.0.1", note = "fn_with_doc")]
    pub fn fn_with_doc() {}

    // @has - '//*[@class="stab deprecated"]' 'Deprecated since 1.0.2: fn_without_doc'
    #[deprecated(since = "1.0.2", note = "fn_without_doc")]
    pub fn fn_without_doc() {}
}

pub trait Bar {
    /// fn_empty_with_doc short
    ///
    /// fn_empty_with_doc full
    #[deprecated(since = "1.0.3", note = "fn_empty_with_doc")]
    fn fn_empty_with_doc();

    #[deprecated(since = "1.0.4", note = "fn_empty_without_doc")]
    fn fn_empty_without_doc();

    /// fn_def_with_doc short
    ///
    /// fn_def_with_doc full
    #[deprecated(since = "1.0.5", note = "fn_def_with_doc")]
    fn fn_def_with_doc() {}

    #[deprecated(since = "1.0.6", note = "fn_def_without_doc")]
    fn fn_def_without_doc() {}

    /// fn_def_def_with_doc short
    ///
    /// fn_def_def_with_doc full
    #[deprecated(since = "1.0.7", note = "fn_def_def_with_doc")]
    fn fn_def_def_with_doc() {}

    #[deprecated(since = "1.0.8", note = "fn_def_def_without_doc")]
    fn fn_def_def_without_doc() {}
}

// @has foo/struct.Foo1.html
pub struct Foo1;

impl Bar for Foo1 {
    // @has - '//*[@class="stab deprecated"]' 'Deprecated since 1.0.3: fn_empty_with_doc'
    // @has - 'fn_empty_with_doc_impl short'
    // @has - 'fn_empty_with_doc_impl full'
    /// fn_empty_with_doc_impl short
    ///
    /// fn_empty_with_doc_impl full
    fn fn_empty_with_doc() {}

    // @has - '//*[@class="stab deprecated"]' 'Deprecated since 1.0.4: fn_empty_without_doc'
    fn fn_empty_without_doc() {}

    // @has - '//*[@class="stab deprecated"]' 'Deprecated since 1.0.5: fn_def_with_doc'
    // @has - 'fn_def_with_doc_impl short'
    // @has - 'fn_def_with_doc_impl full'
    /// fn_def_with_doc_impl short
    ///
    /// fn_def_with_doc_impl full
    fn fn_def_with_doc() {}

    // @has - '//*[@class="stab deprecated"]' 'Deprecated since 1.0.6: fn_def_without_doc'
    fn fn_def_without_doc() {}

    // @has - '//*[@class="stab deprecated"]' 'Deprecated since 1.0.7: fn_def_def_with_doc'
    // @has - 'fn_def_def_with_doc short'
    // @!has - 'fn_def_def_with_doc full'

    // @has - '//*[@class="stab deprecated"]' 'Deprecated since 1.0.8: fn_def_def_without_doc'
}

// @has foo/struct.Foo2.html
pub struct Foo2;

impl Bar for Foo2 {
    // @has - '//*[@class="stab deprecated"]' 'Deprecated since 1.0.3: fn_empty_with_doc'
    // @has - 'fn_empty_with_doc short'
    // @!has - 'fn_empty_with_doc full'
    fn fn_empty_with_doc() {}

    // @has - '//*[@class="stab deprecated"]' 'Deprecated since 1.0.4: fn_empty_without_doc'
    // @has - 'fn_empty_without_doc_impl short'
    // @has - 'fn_empty_without_doc_impl full'
    /// fn_empty_without_doc_impl short
    ///
    /// fn_empty_without_doc_impl full
    fn fn_empty_without_doc() {}

    // @has - '//*[@class="stab deprecated"]' 'Deprecated since 1.0.5: fn_def_with_doc'
    // @has - 'fn_def_with_doc short'
    // @!has - 'fn_def_with_doc full'
    fn fn_def_with_doc() {}

    // @has - '//*[@class="stab deprecated"]' 'Deprecated since 1.0.6: fn_def_without_doc'
    // @has - 'fn_def_without_doc_impl short'
    // @has - 'fn_def_without_doc_impl full'
    /// fn_def_without_doc_impl short
    ///
    /// fn_def_without_doc_impl full
    fn fn_def_without_doc() {}

    // @has - '//*[@class="stab deprecated"]' 'Deprecated since 1.0.7: fn_def_def_with_doc'
    // @has - 'fn_def_def_with_doc short'
    // @!has - 'fn_def_def_with_doc full'

    // @has - '//*[@class="stab deprecated"]' 'Deprecated since 1.0.8: fn_def_def_without_doc'
}
