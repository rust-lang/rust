//@ edition: 2021

extern "C" {
    //@ is '$.index[?(@.name=="A")].inner.static.is_unsafe'  true
    //@ is '$.index[?(@.name=="A")].inner.static.is_mutable' false
    pub static A: i32;
    //@ is '$.index[?(@.name=="B")].inner.static.is_unsafe'  true
    //@ is '$.index[?(@.name=="B")].inner.static.is_mutable' true
    pub static mut B: i32;

    // items in unadorned `extern` blocks cannot have safety qualifiers
}

unsafe extern "C" {
    //@ is '$.index[?(@.name=="C")].inner.static.is_unsafe'  true
    //@ is '$.index[?(@.name=="C")].inner.static.is_mutable' false
    pub static C: i32;
    //@ is '$.index[?(@.name=="D")].inner.static.is_unsafe'  true
    //@ is '$.index[?(@.name=="D")].inner.static.is_mutable' true
    pub static mut D: i32;

    //@ is '$.index[?(@.name=="E")].inner.static.is_unsafe'  false
    //@ is '$.index[?(@.name=="E")].inner.static.is_mutable' false
    pub safe static E: i32;
    //@ is '$.index[?(@.name=="F")].inner.static.is_unsafe'  false
    //@ is '$.index[?(@.name=="F")].inner.static.is_mutable' true
    pub safe static mut F: i32;

    //@ is '$.index[?(@.name=="G")].inner.static.is_unsafe'  true
    //@ is '$.index[?(@.name=="G")].inner.static.is_mutable' false
    pub unsafe static G: i32;
    //@ is '$.index[?(@.name=="H")].inner.static.is_unsafe'  true
    //@ is '$.index[?(@.name=="H")].inner.static.is_mutable' true
    pub unsafe static mut H: i32;
}

//@ ismany '$.index[?(@.inner.static)].inner.static.expr' '""' '""' '""' '""' '""' '""' '""' '""'
//@ ismany '$.index[?(@.inner.static)].inner.static.type.primitive' '"i32"' '"i32"' '"i32"' '"i32"' '"i32"' '"i32"' '"i32"' '"i32"'
