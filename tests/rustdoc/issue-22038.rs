extern "C" {
    // @has issue_22038/fn.foo1.html \
    //      '//div[@class="item-decl"]/pre[@class="rust"]' 'pub unsafe extern "C" fn foo1()'
    pub fn foo1();
}

extern "system" {
    // @has issue_22038/fn.foo2.html \
    //      '//div[@class="item-decl"]/pre[@class="rust"]' 'pub unsafe extern "system" fn foo2()'
    pub fn foo2();
}

// @has issue_22038/fn.bar.html \
//      '//div[@class="item-decl"]/pre[@class="rust"]' 'pub extern "C" fn bar()'
pub extern "C" fn bar() {}

// @has issue_22038/fn.baz.html \
//      '//div[@class="item-decl"]/pre[@class="rust"]' 'pub extern "system" fn baz()'
pub extern "system" fn baz() {}
