// First a module with inner documentation

// @has issue_55364/subone/index.html
// These foo/bar links in the module's documentation should refer inside `subone`
// @has - '//section[@id="main-content"]/details[@open=""]/div[@class="docblock"]//a[@href="fn.foo.html"]' 'foo'
// @has - '//section[@id="main-content"]/details[@open=""]/div[@class="docblock"]//a[@href="fn.bar.html"]' 'bar'
pub mod subone {
    //! See either [foo] or [bar].

    // This should refer to subone's `bar`
    // @has issue_55364/subone/fn.foo.html
    // @has - '//section[@id="main-content"]/details/div[@class="docblock"]//a[@href="fn.bar.html"]' 'bar'
    /// See [bar]
    pub fn foo() {}
    // This should refer to subone's `foo`
    // @has issue_55364/subone/fn.bar.html
    // @has - '//section[@id="main-content"]/details/div[@class="docblock"]//a[@href="fn.foo.html"]' 'foo'
    /// See [foo]
    pub fn bar() {}
}

// A module with outer documentation

// @has issue_55364/subtwo/index.html
// These foo/bar links in the module's documentation should not reference inside `subtwo`
// @!has - '//section[@id="main-content"]/div[@class="docblock"]//a[@href="fn.foo.html"]' 'foo'
// @!has - '//section[@id="main-content"]/div[@class="docblock"]//a[@href="fn.bar.html"]' 'bar'
// Instead it should be referencing the top level functions
// @has - '//section[@id="main-content"]/details/div[@class="docblock"]//a[@href="../fn.foo.html"]' 'foo'
// @has - '//section[@id="main-content"]/details/div[@class="docblock"]//a[@href="../fn.bar.html"]' 'bar'
// Though there should be such links later
// @has - '//section[@id="main-content"]/ul[@class="item-table"]//div[@class="item-name"]/a[@class="fn"][@href="fn.foo.html"]' 'foo'
// @has - '//section[@id="main-content"]/ul[@class="item-table"]//div[@class="item-name"]/a[@class="fn"][@href="fn.bar.html"]' 'bar'
/// See either [foo] or [bar].
pub mod subtwo {

    // Despite the module's docs referring to the top level foo/bar,
    // this should refer to subtwo's `bar`
    // @has issue_55364/subtwo/fn.foo.html
    // @has - '//section[@id="main-content"]/details/div[@class="docblock"]//a[@href="fn.bar.html"]' 'bar'
    /// See [bar]
    pub fn foo() {}
    // Despite the module's docs referring to the top level foo/bar,
    // this should refer to subtwo's `foo`
    // @has issue_55364/subtwo/fn.bar.html
    // @has - '//section[@id="main-content"]/details/div[@class="docblock"]//a[@href="fn.foo.html"]' 'foo'
    /// See [foo]
    pub fn bar() {}
}

// These are the function referred to by the module above with outer docs

/// See [bar]
pub fn foo() {}
/// See [foo]
pub fn bar() {}

// This module refers to the outer foo/bar by means of `super::`

// @has issue_55364/subthree/index.html
// This module should also refer to the top level foo/bar
// @has - '//section[@id="main-content"]/details/div[@class="docblock"]//a[@href="../fn.foo.html"]' 'foo'
// @has - '//section[@id="main-content"]/details/div[@class="docblock"]//a[@href="../fn.bar.html"]' 'bar'
pub mod subthree {
    //! See either [foo][super::foo] or [bar][super::bar]
}

// Next we go *deeper* - In order to ensure it's not just "this or parent"
// we test `crate::` and a `super::super::...` chain
// @has issue_55364/subfour/subfive/subsix/subseven/subeight/index.html
// @has - '//section[@id="main-content"]/ul[@class="item-table"]//div[@class="desc docblock-short"]//a[@href="../../../../../subone/fn.foo.html"]' 'other foo'
// @has - '//section[@id="main-content"]/ul[@class="item-table"]//div[@class="desc docblock-short"]//a[@href="../../../../../subtwo/fn.bar.html"]' 'other bar'
pub mod subfour {
    pub mod subfive {
        pub mod subsix {
            pub mod subseven {
                pub mod subeight {
                    /// See [other foo][crate::subone::foo]
                    pub fn foo() {}
                    /// See [other bar][super::super::super::super::super::subtwo::bar]
                    pub fn bar() {}
                }
            }
        }
    }
}
