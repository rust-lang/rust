// rustfmt-normalize_comments: true
// rustfmt-wrap_comments: true

/// A Doc comment
#[AnAttribute]
pub struct Foo {
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

// Destructuring
fn foo() {
    S { x: 5, .. };
    Struct { .. } = Struct { a: 1, b: 4 };
    Struct { a, .. } = Struct { a: 1, b: 2, c: 3 };
    TupleStruct(a, .., b) = TupleStruct(1, 2);
    TupleStruct(..) = TupleStruct(3, 4);
    TupleStruct(
        aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa,
        ..,
        bbbbbbbbbbbbbbbbbbbbbbbbbbbbbb,
    ) = TupleStruct(1, 2);
}

// #1095
struct S<T /* comment */> {
    t: T,
}

// #1029
pub struct Foo {
    #[doc(hidden)]
    // This will NOT get deleted!
    bar: String, // hi
}

// #1029
struct X {
    // `x` is an important number.
    #[allow(unused)] // TODO: use
    x: u32,
}

// #410
#[allow(missing_docs)]
pub struct Writebatch<K: Key> {
    #[allow(dead_code)] // only used for holding the internal pointer
    writebatch: RawWritebatch,
    marker: PhantomData<K>,
}

struct Bar;

struct NewType(Type, OtherType);

struct NewInt<T: Copy>(
    pub i32,
    SomeType, // inline comment
    T,        // sup
);

struct Qux<
    'a,
    N: Clone + 'a,
    E: Clone + 'a,
    G: Labeller<'a, N, E> + GraphWalk<'a, N, E>,
    W: Write + Copy,
>(
    AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA, // Comment
    BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB,
    #[AnAttr]
    // Comment
    /// Testdoc
    G,
    pub W,
);

struct Tuple(
    // Comment 1
    AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA,
    // Comment 2
    BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB,
);

// With a where-clause and generics.
pub struct Foo<'a, Y: Baz>
where
    X: Whatever,
{
    f: SomeType, // Comment beside a field
}

struct Baz {
    a: A, // Comment A
    b: B, // Comment B
    c: C, // Comment C
}

struct Baz {
    a: A, // Comment A

    b: B, // Comment B

    c: C, // Comment C
}

struct Baz {
    a: A,

    b: B,
    c: C,

    d: D,
}

struct Baz {
    // Comment A
    a: A,

    // Comment B
    b: B,
    // Comment C
    c: C,
}

// Will this be a one-liner?
struct Tuple(
    A, // Comment
    B,
);

pub struct State<F: FnMut() -> time::Timespec> {
    now: F,
}

pub struct State<F: FnMut() -> ()> {
    now: F,
}

pub struct State<F: FnMut()> {
    now: F,
}

struct Palette {
    /// A map of indices in the palette to a count of pixels in approximately
    /// that color
    foo: i32,
}

// Splitting a single line comment into a block previously had a misalignment
// when the field had attributes
struct FieldsWithAttributes {
    // Pre Comment
    #[rustfmt::skip] pub host:String, /* Post comment BBBBBBBBBBBBBB BBBBBBBBBBBBBBBB
                                       * BBBBBBBBBBBBBBBB BBBBBBBBBBBBBBBBB BBBBBBBBBBB */
    // Another pre comment
    #[attr1]
    #[attr2]
    pub id: usize, /* CCCCCCCCCCCCCCCCCCC CCCCCCCCCCCCCCCCCCC CCCCCCCCCCCCCCCC
                    * CCCCCCCCCCCCCCCCCC CCCCCCCCCCCCCC CCCCCCCCCCCC */
}

struct Deep {
    deeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeep:
        node::Handle<IdRef<'id, Node<K, V>>, Type, NodeType>,
}

struct Foo<T>(T);
struct Foo<T>(T)
where
    T: Copy,
    T: Eq;
struct Foo<T>(
    TTTTTTTTTTTTTTTTT,
    UUUUUUUUUUUUUUUUUUUUUUUU,
    TTTTTTTTTTTTTTTTTTT,
    UUUUUUUUUUUUUUUUUUU,
);
struct Foo<T>(
    TTTTTTTTTTTTTTTTTT,
    UUUUUUUUUUUUUUUUUUUUUUUU,
    TTTTTTTTTTTTTTTTTTT,
)
where
    T: PartialEq;
struct Foo<T>(
    TTTTTTTTTTTTTTTTT,
    UUUUUUUUUUUUUUUUUUUUUUUU,
    TTTTTTTTTTTTTTTTTTTTT,
)
where
    T: PartialEq;
struct Foo<T>(
    TTTTTTTTTTTTTTTTT,
    UUUUUUUUUUUUUUUUUUUUUUUU,
    TTTTTTTTTTTTTTTTTTT,
    UUUUUUUUUUUUUUUUUUU,
)
where
    T: PartialEq;
struct Foo<T>(
    TTTTTTTTTTTTTTTTT,        // Foo
    UUUUUUUUUUUUUUUUUUUUUUUU, // Bar
    // Baz
    TTTTTTTTTTTTTTTTTTT,
    // Qux (FIXME #572 - doc comment)
    UUUUUUUUUUUUUUUUUUU,
);

mod m {
    struct X<T>
    where
        T: Sized,
    {
        a: T,
    }
}

struct Foo<T>(
    TTTTTTTTTTTTTTTTTTT,
    /// Qux
    UUUUUUUUUUUUUUUUUUU,
);

struct Issue677 {
    pub ptr: *const libc::c_void,
    pub trace: fn(obj: *const libc::c_void, tracer: *mut JSTracer),
}

struct Foo {}
struct Foo {}
struct Foo {
    // comment
}
struct Foo {
    // trailing space ->
}
struct Foo {
    // comment
}
struct Foo(
    // comment
);

struct LongStruct {
    a: A,
    the_quick_brown_fox_jumps_over_the_lazy_dog:
        AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA,
}

struct Deep {
    deeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeep:
        node::Handle<IdRef<'id, Node<Key, Value>>, Type, NodeType>,
}

struct Foo<C = ()>(String);

// #1364
fn foo() {
    convex_shape.set_point(0, &Vector2f { x: 400.0, y: 100.0 });
    convex_shape.set_point(1, &Vector2f { x: 500.0, y: 70.0 });
    convex_shape.set_point(2, &Vector2f { x: 450.0, y: 100.0 });
    convex_shape.set_point(3, &Vector2f { x: 580.0, y: 150.0 });
}

// Vertical alignment
struct Foo {
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

// structs with long identifier
struct Loooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooong
{}
struct Looooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooong
{}
struct Loooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooong
{}
struct Loooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooong
{
    x: i32,
}

// structs with visibility, do not duplicate visibility (#2110).
pub(self) struct Foo {}
pub(super) struct Foo {}
pub(crate) struct Foo {}
pub(self) struct Foo();
pub(super) struct Foo();
pub(crate) struct Foo();

// #2125
pub struct ReadinessCheckRegistry(
    Mutex<HashMap<Arc<String>, Box<Fn() -> ReadinessCheck + Sync + Send>>>,
);

// #2144 unit struct with generics
struct MyBox<T: ?Sized>;
struct MyBoxx<T, S>
where
    T: ?Sized,
    S: Clone;

// #2208
struct Test {
    /// foo
    #[serde(default)]
    pub join: Vec<String>,
    #[serde(default)]
    pub tls: bool,
}

// #2818
struct Paren((i32))
where
    i32: Trait;
struct Parens((i32, i32))
where
    i32: Trait;
