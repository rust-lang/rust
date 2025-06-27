extern "C" {
    //@ jq .index[] | select(.name == "f1").inner.function.header?.is_unsafe == true
    pub fn f1();

    // items in `extern` blocks without an `unsafe` qualifier cannot have safety qualifiers
}

unsafe extern "C" {
    //@ jq .index[] | select(.name == "f4").inner.function.header?.is_unsafe == true
    pub fn f4();

    //@ jq .index[] | select(.name == "f5").inner.function.header?.is_unsafe == true
    pub unsafe fn f5();

    //@ jq .index[] | select(.name == "f6").inner.function.header?.is_unsafe == false
    pub safe fn f6();
}
