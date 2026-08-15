// #5215
struct MyTuple(
    /// Doc Comments
    /* TODO note to add more to Doc Comments */
    u32,
    /// Doc Comments
    // TODO note
    u64,
);

struct MyTuple(
    #[cfg(unix)] // some comment
    u64,
    #[cfg(not(unix))] /*block comment */ u32,
);

struct MyTuple(
    #[cfg(unix)]
    // some comment
    u64,
    #[cfg(not(unix))]
    /*block comment */
    u32,
);

struct MyTuple(
    #[cfg(unix)] // some comment
    pub  u64,
    #[cfg(not(unix))] /*block comment */ pub(crate)  u32,
);

struct MyTuple(
    /// Doc Comments
    /* TODO note to add more to Doc Comments */
    pub  u32,
    /// Doc Comments
    // TODO note
    pub(crate)  u64,
);

struct MyStruct {
    #[cfg(unix)] // some comment
    a: u64,
    #[cfg(not(unix))] /*block comment */ b: u32,
}

struct MyStruct {
    #[cfg(unix)] // some comment
    pub a: u64,
    #[cfg(not(unix))] /*block comment */ pub(crate) b: u32,
}

struct MyStruct {
    /// Doc Comments
    /* TODO note to add more to Doc Comments */
    a: u32,
    /// Doc Comments
    // TODO note
    b: u64,
}

struct MyStruct {
    /// Doc Comments
    /* TODO note to add more to Doc Comments */
    pub a: u32,
    /// Doc Comments
    // TODO note
    pub(crate) b: u64,
}
