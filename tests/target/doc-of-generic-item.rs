// Non-doc pre-comment of Foo
/// doc of Foo
// Non-doc post-comment of Foo
struct Foo<
    // Non-doc pre-comment of 'a
    /// doc of 'a
    'a,
    // Non-doc pre-comment of T
    /// doc of T
    T,
    // Non-doc pre-comment of N
    /// doc of N
    const N: item,
>;
