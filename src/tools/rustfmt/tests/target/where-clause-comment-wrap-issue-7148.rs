//@ rustfmt-wrap_comments: true
//@ rustfmt-max_width: 100
//@ rustfmt-comment_width: 100

// Comments between `where` and the first predicate should be formatted using
// the full column budget rather than a budget derived from the header width.
impl<T> SomeTrait for LongTypeName<T>
where
    // A long comment line between where and the first clause that fits within max width limit
    // already. So no additional wrapping should occur.
    T: SomeTrait,

    // A long comment line between where and the first clause that fits within max width limit
    // already. So no additional wrapping should occur.
    T: OtherTrait,
{
}

// Comments between `where` and the first predicate should be formatted using
// the full column budget rather than a budget derived from the header width.
impl<T> SomeOtherTrait for LongTypeName<T>
where
    // A long comment line between where and the first clause that doesn't fit within max width
    // limit already. So additional wrapping should occur.
    T: SomeTrait,

    // A long comment line between where and the first clause that doesn't fit within max width
    // limit already. So additional wrapping should occur.
    T: OtherTrait,
{
}
