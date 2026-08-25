// Test to ensure the feature is working as expected.

#![crate_name = "foo"]

//@ has 'foo/index.html'

// First we check that both the static and the function have a "sup" element
// to tell they're unsafe.

//@ count - '//dl[@class="item-table"]//sup[@title="unsafe static"]' 1
//@ has - '//dl[@class="item-table"]//sup[@title="unsafe static"]' '⚠'
//@ count - '//dl[@class="item-table"]//sup[@title="unsafe function"]' 1
//@ has - '//dl[@class="item-table"]//sup[@title="unsafe function"]' '⚠'

unsafe extern "C" {
    //@ has 'foo/static.FOO.html'
    //@ has - '//pre[@class="rust item-decl"]' 'pub static FOO: i32'
    pub safe static FOO: i32;
    //@ has 'foo/static.BAR.html'
    //@ has - '//pre[@class="rust item-decl"]' 'pub unsafe static BAR: i32'
    pub static BAR: i32;

    //@ has 'foo/fn.foo.html'
    //@ has - '//pre[@class="rust item-decl"]' 'pub extern "C" fn foo()'
    pub safe fn foo();
    //@ has 'foo/fn.bar.html'
    //@ has - '//pre[@class="rust item-decl"]' 'pub unsafe extern "C" fn bar()'
    pub fn bar();
}
