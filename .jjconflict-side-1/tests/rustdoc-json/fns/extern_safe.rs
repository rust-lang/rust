extern "C" {
    //@ is "$.index[?(@.name=='f1')].inner.function.header.is_unsafe" true
    pub fn f1();

    // items in `extern` blocks without an `unsafe` qualifier cannot have safety qualifiers
}

unsafe extern "C" {
    //@ is "$.index[?(@.name=='f4')].inner.function.header.is_unsafe" true
    pub fn f4();

    //@ is "$.index[?(@.name=='f5')].inner.function.header.is_unsafe" true
    pub unsafe fn f5();

    //@ is "$.index[?(@.name=='f6')].inner.function.header.is_unsafe" false
    pub safe fn f6();
}
