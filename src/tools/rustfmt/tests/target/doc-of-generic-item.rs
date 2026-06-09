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

// Non-doc pre-comment of Foo
/// doc of Foo
// Non-doc post-comment of Foo
struct Foo<
    // Non-doc pre-comment of 'a
    /// doc of 'a
    // Non-doc post-comment of 'a
    'a,
    // Non-doc pre-comment of T
    /// doc of T
    // Non-doc post-comment of T
    T,
    // Non-doc pre-comment of N
    /// doc of N
    // Non-doc post-comment of N
    const N: item,
>;
