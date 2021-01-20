extern {
    // @has issue_22038/fn.foo1.html \
    //      '//*[@class="rust fn"]' 'pub unsafe extern "C" fn foo1()'
    pub fn foo1();
}

extern "system" {
    // @has issue_22038/fn.foo2.html \
    //      '//*[@class="rust fn"]' 'pub unsafe extern "system" fn foo2()'
    pub fn foo2();
}

// @has issue_22038/fn.bar.html \
//      '//*[@class="rust fn"]' 'pub extern "C" fn bar()'
pub extern fn bar() {}

// @has issue_22038/fn.baz.html \
//      '//*[@class="rust fn"]' 'pub extern "system" fn baz()'
pub extern "system" fn baz() {}
