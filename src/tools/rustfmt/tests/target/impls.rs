// rustfmt-normalize_comments: true
impl Foo for Bar {
    fn foo() {
        "hi"
    }
}

pub impl Foo for Bar {
    // Associated Constants
    const Baz: i32 = 16;
    // Associated Types
    type FooBar = usize;
    // Comment 1
    fn foo() {
        "hi"
    }
    // Comment 2
    fn foo() {
        "hi"
    }
    // Comment 3
}

#[inherent]
impl Visible for Bar {
    pub const C: i32;
    pub type T;
    pub fn f();
    pub fn g() {}
}

pub unsafe impl<'a, 'b, X, Y: Foo<Bar>> !Foo<'a, X> for Bar<'b, Y>
where
    X: Foo<'a, Z>,
{
    fn foo() {
        "hi"
    }
}

impl<'a, 'b, X, Y: Foo<Bar>> Foo<'a, X> for Bar<'b, Y>
where
    X: Fooooooooooooooooooooooooooooo<'a, Z>,
{
    fn foo() {
        "hi"
    }
}

impl<'a, 'b, X, Y: Foo<Bar>> Foo<'a, X> for Bar<'b, Y>
where
    X: Foooooooooooooooooooooooooooo<'a, Z>,
{
    fn foo() {
        "hi"
    }
}

impl<T> Foo for Bar<T> where T: Baz {}

impl<T> Foo for Bar<T>
where
    T: Baz,
{
    // Comment
}

impl Foo {
    fn foo() {}
}

impl Boo {
    // BOO
    fn boo() {}
    // FOO
}

mod a {
    impl Foo {
        // Hello!
        fn foo() {}
    }
}

mod b {
    mod a {
        impl Foo {
            fn foo() {}
        }
    }
}

impl Foo {
    add_fun!();
}

impl Blah {
    fn boop() {}
    add_fun!();
}

impl X {
    fn do_parse(mut self: X) {}
}

impl Y5000 {
    fn bar(self: X<'a, 'b>, y: Y) {}

    fn bad(&self, (x, y): CoorT) {}

    fn turbo_bad(self: X<'a, 'b>, (x, y): CoorT) {}
}

pub impl<T> Foo for Bar<T>
where
    T: Foo,
{
    fn foo() {
        "hi"
    }
}

pub impl<T, Z> Foo for Bar<T, Z>
where
    T: Foo,
    Z: Baz,
{
}

mod m {
    impl<T> PartialEq for S<T>
    where
        T: PartialEq,
    {
        fn eq(&self, other: &Self) {
            true
        }
    }

    impl<T> PartialEq for S<T> where T: PartialEq {}
}

impl<BorrowType, K, V, NodeType, HandleType>
    Handle<NodeRef<BorrowType, K, V, NodeType>, HandleType>
{
}

impl<BorrowType, K, V, NodeType, HandleType> PartialEq
    for Handle<NodeRef<BorrowType, K, V, NodeType>, HandleType>
{
}

mod x {
    impl<A, B, C, D> Foo
    where
        A: 'static,
        B: 'static,
        C: 'static,
        D: 'static,
    {
    }
}

impl<ConcreteThreadSafeLayoutNode: ThreadSafeLayoutNodeFoo>
    Issue1249<ConcreteThreadSafeLayoutNode>
{
    // Creates a new flow constructor.
    fn foo() {}
}

// #1600
impl<#[may_dangle] K, #[may_dangle] V> Drop for RawTable<K, V> {
    fn drop() {}
}

// #1168
pub trait Number:
    Copy
    + Eq
    + Not<Output = Self>
    + Shl<u8, Output = Self>
    + Shr<u8, Output = Self>
    + BitAnd<Self, Output = Self>
    + BitOr<Self, Output = Self>
    + BitAndAssign
    + BitOrAssign
{
    // test
    fn zero() -> Self;
}

// #1642
pub trait SomeTrait:
    Clone
    + Eq
    + PartialEq
    + Ord
    + PartialOrd
    + Default
    + Hash
    + Debug
    + Display
    + Write
    + Read
    + FromStr
{
    // comment
}

// #1995
impl Foo {
    fn f(
        S {
            aaaaaaaaaa: aaaaaaaaaa,
            bbbbbbbbbb: bbbbbbbbbb,
            cccccccccc: cccccccccc,
        }: S,
    ) -> u32 {
        1
    }
}

// #2491
impl<'a, 'b, 'c> SomeThing<Something>
    for (
        &'a mut SomethingLong,
        &'b mut SomethingLong,
        &'c mut SomethingLong,
    )
{
    fn foo() {}
}

// #2746
impl<'seq1, 'seq2, 'body, 'scope, Channel>
    Adc12<
        Dual,
        MasterRunningDma<'seq1, 'body, 'scope, Channel>,
        SlaveRunningDma<'seq2, 'body, 'scope>,
    >
where
    Channel: DmaChannel,
{
}

// #4084
impl const std::default::Default for Struct {
    #[inline]
    fn default() -> Self {
        Self { f: 12.5 }
    }
}
