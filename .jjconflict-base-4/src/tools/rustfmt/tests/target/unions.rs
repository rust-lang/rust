// rustfmt-normalize_comments: true
// rustfmt-wrap_comments: true

/// A Doc comment
#[AnAttribute]
pub union Foo {
    #[rustfmt::skip]
    f :   SomeType, // Comment beside a field
    f: SomeType, // Comment beside a field
    // Comment on a field
    #[AnAttribute]
    g: SomeOtherType,
    /// A doc comment on a field
    h: AThirdType,
    pub i: TypeForPublicField,
}

// #1029
pub union Foo {
    #[doc(hidden)]
    // This will NOT get deleted!
    bar: String, // hi
}

// #1029
union X {
    // `x` is an important number.
    #[allow(unused)] // TODO: use
    x: u32,
}

// #410
#[allow(missing_docs)]
pub union Writebatch<K: Key> {
    #[allow(dead_code)] // only used for holding the internal pointer
    writebatch: RawWritebatch,
    marker: PhantomData<K>,
}

// With a where-clause and generics.
pub union Foo<'a, Y: Baz>
where
    X: Whatever,
{
    f: SomeType, // Comment beside a field
}

union Baz {
    a: A, // Comment A
    b: B, // Comment B
    c: C, // Comment C
}

union Baz {
    a: A, // Comment A

    b: B, // Comment B

    c: C, // Comment C
}

union Baz {
    a: A,

    b: B,
    c: C,

    d: D,
}

union Baz {
    // Comment A
    a: A,

    // Comment B
    b: B,
    // Comment C
    c: C,
}

pub union State<F: FnMut() -> time::Timespec> {
    now: F,
}

pub union State<F: FnMut() -> ()> {
    now: F,
}

pub union State<F: FnMut()> {
    now: F,
}

union Palette {
    /// A map of indices in the palette to a count of pixels in approximately
    /// that color
    foo: i32,
}

// Splitting a single line comment into a block previously had a misalignment
// when the field had attributes
union FieldsWithAttributes {
    // Pre Comment
    #[rustfmt::skip] pub host:String, /* Post comment BBBBBBBBBBBBBB BBBBBBBBBBBBBBBB
                                       * BBBBBBBBBBBBBBBB BBBBBBBBBBBBBBBBB BBBBBBBBBBB */
    // Another pre comment
    #[attr1]
    #[attr2]
    pub id: usize, /* CCCCCCCCCCCCCCCCCCC CCCCCCCCCCCCCCCCCCC CCCCCCCCCCCCCCCC
                    * CCCCCCCCCCCCCCCCCC CCCCCCCCCCCCCC CCCCCCCCCCCC */
}

union Deep {
    deeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeep:
        node::Handle<IdRef<'id, Node<K, V>>, Type, NodeType>,
}

mod m {
    union X<T>
    where
        T: Sized,
    {
        a: T,
    }
}

union Issue677 {
    pub ptr: *const libc::c_void,
    pub trace: fn(obj: *const libc::c_void, tracer: *mut JSTracer),
}

union Foo {}
union Foo {}
union Foo {
    // comment
}
union Foo {
    // trailing space ->
}
union Foo {
    // comment
}

union LongUnion {
    a: A,
    the_quick_brown_fox_jumps_over_the_lazy_dog:
        AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA,
}

union Deep {
    deeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeep:
        node::Handle<IdRef<'id, Node<Key, Value>>, Type, NodeType>,
}

// #1364
fn foo() {
    convex_shape.set_point(0, &Vector2f { x: 400.0, y: 100.0 });
    convex_shape.set_point(1, &Vector2f { x: 500.0, y: 70.0 });
    convex_shape.set_point(2, &Vector2f { x: 450.0, y: 100.0 });
    convex_shape.set_point(3, &Vector2f { x: 580.0, y: 150.0 });
}

// Vertical alignment
union Foo {
    aaaaa: u32, // a

    b: u32,  // b
    cc: u32, // cc

    xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx: u32, // 1
    yy: u32,  // comment2
    zzz: u32, // comment3

    aaaaaa: u32, // comment4
    bb: u32,     // comment5
    // separate
    dd: u32, // comment7
    c: u32,  // comment6

    aaaaaaa: u32, /* multi
                   * line
                   * comment
                   */
    b: u32, // hi

    do_not_push_this_comment1: u32, // comment1
    xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx: u32, // 2
    please_do_not_push_this_comment3: u32, // comment3

    do_not_push_this_comment1: u32, // comment1
    // separate
    xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx: u32, // 2
    please_do_not_push_this_comment3: u32, // comment3

    do_not_push_this_comment1: u32, // comment1
    xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx: u32, // 2
    // separate
    please_do_not_push_this_comment3: u32, // comment3
}
