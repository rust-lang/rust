// rustfmt-wrap_comments: true

                                                                       /// A Doc comment
#[AnAttribute]
pub struct Foo {
    #[rustfmt_skip]
    f :   SomeType, // Comment beside a field
    f: SomeType, // Comment beside a field
    // Comment on a field
    #[AnAttribute]
    g: SomeOtherType,
      /// A doc comment on a field
    h: AThirdType,
    pub i: TypeForPublicField
}

struct Bar;

struct NewType(Type,       OtherType);

struct
NewInt     <T: Copy>(pub i32, SomeType /* inline comment */, T /* sup */


    );

struct Qux<'a,
           N: Clone + 'a,
           E: Clone + 'a,
           G: Labeller<'a, N, E> + GraphWalk<'a, N, E>,
           W: Write + Copy>
(
    AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA, // Comment
    BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB,
    #[AnAttr]
    // Comment
    /// Testdoc
    G,
    pub W,
);

struct Tuple(/*Comment 1*/ AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA,
             /* Comment 2   */ BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB,);

// With a where clause and generics.
pub struct Foo<'a, Y: Baz>
    where X: Whatever
{
    f: SomeType, // Comment beside a field
}

struct Baz {

    a: A,  // Comment A
    b: B, // Comment B
    c: C,   // Comment C

}

struct Baz {
    a: A,  // Comment A

    b: B, // Comment B




    c: C,   // Comment C
}

struct Baz {

    a: A,

    b: B,
    c: C,



    
    d: D

}

struct Baz
{
    // Comment A
    a: A,
    
    // Comment B
b: B,
    // Comment C
      c: C,}

// Will this be a one-liner?
struct Tuple(
    A, //Comment
    B
);

pub struct State<F: FnMut() -> time::Timespec> { now: F }

pub struct State<F: FnMut() -> ()> { now: F }

pub struct State<F: FnMut()> { now: F }

struct Palette { /// A map of indizes in the palette to a count of pixels in approximately that color
                    foo: i32}

// Splitting a single line comment into a block previously had a misalignment
// when the field had attributes
struct FieldsWithAttributes {
    // Pre Comment
    #[rustfmt_skip] pub host:String, // Post comment BBBBBBBBBBBBBB BBBBBBBBBBBBBBBB BBBBBBBBBBBBBBBB BBBBBBBBBBBBBBBBB BBBBBBBBBBB
    //Another pre comment
    #[attr1]
    #[attr2] pub id: usize // CCCCCCCCCCCCCCCCCCC CCCCCCCCCCCCCCCCCCC CCCCCCCCCCCCCCCC CCCCCCCCCCCCCCCCCC CCCCCCCCCCCCCC CCCCCCCCCCCC
}

struct Deep {
    deeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeep: node::Handle<IdRef<'id, Node<K, V>>,
                                                     Type,
                                                     NodeType>,
}

struct Foo<T>(T);
struct Foo<T>(T) where T: Copy, T: Eq;
struct Foo<T>(TTTTTTTTTTTTTTTTT, UUUUUUUUUUUUUUUUUUUUUUUU, TTTTTTTTTTTTTTTTTTT, UUUUUUUUUUUUUUUUUUU);
struct Foo<T>(TTTTTTTTTTTTTTTTTT, UUUUUUUUUUUUUUUUUUUUUUUU, TTTTTTTTTTTTTTTTTTT) where T: PartialEq;
struct Foo<T>(TTTTTTTTTTTTTTTTT, UUUUUUUUUUUUUUUUUUUUUUUU, TTTTTTTTTTTTTTTTTTTTT) where T: PartialEq;
struct Foo<T>(TTTTTTTTTTTTTTTTT, UUUUUUUUUUUUUUUUUUUUUUUU, TTTTTTTTTTTTTTTTTTT, UUUUUUUUUUUUUUUUUUU) where T: PartialEq;
struct Foo<T>(TTTTTTTTTTTTTTTTT, // Foo
              UUUUUUUUUUUUUUUUUUUUUUUU /* Bar */,
              // Baz
              TTTTTTTTTTTTTTTTTTT,
              // Qux (FIXME #572 - doc comment)
              UUUUUUUUUUUUUUUUUUU);

mod m {
    struct X<T> where T: Sized {
        a: T,
    }
}

struct Foo<T>(TTTTTTTTTTTTTTTTTTT,
              /// Qux
    UUUUUUUUUUUUUUUUUUU);

struct Issue677 {
    pub ptr: *const libc::c_void,
    pub trace: fn(  obj: 
          *const libc::c_void, tracer   : *mut   JSTracer   ),
}
