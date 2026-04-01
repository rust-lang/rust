// rustfmt-struct_field_align_threshold: 20
// rustfmt-normalize_comments: true
// rustfmt-wrap_comments: true
// rustfmt-error_on_line_overflow: false

struct Foo {
    x: u32,
    yy: u32, // comment
    zzz: u32,
}

pub struct Bar {
    x: u32,
    yy: u32,
    zzz: u32,

    xxxxxxx: u32,
}

fn main() {
    let foo = Foo {
        x: 0,
        yy: 1,
        zzz: 2,
    };

    let bar = Bar {
        x: 0,
        yy: 1,
        zzz: 2,

        xxxxxxx: 3,
    };
}

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
    pub i: TypeForPublicField
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
    #[allow(dead_code)] //only used for holding the internal pointer
    writebatch: RawWritebatch,
    marker: PhantomData<K>,
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

// With a where-clause and generics.
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

struct Palette { /// A map of indices in the palette to a count of pixels in approximately that color
                    foo: i32}

// Splitting a single line comment into a block previously had a misalignment
// when the field had attributes
struct FieldsWithAttributes {
    // Pre Comment
    #[rustfmt::skip] pub host:String, // Post comment BBBBBBBBBBBBBB BBBBBBBBBBBBBBBB BBBBBBBBBBBBBBBB BBBBBBBBBBBBBBBBB BBBBBBBBBBB
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

struct Foo {}
struct Foo {
    }
struct Foo {
    // comment
    }
struct Foo {
    // trailing space ->    


    }
struct Foo { /* comment */ }
struct Foo( /* comment */ );

struct LongStruct {
    a: A,
    the_quick_brown_fox_jumps_over_the_lazy_dog:AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA,
}

struct Deep {
    deeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeep: node::Handle<IdRef<'id, Node<Key, Value>>,
                                                                         Type,
                                                                         NodeType>,
}

struct Foo<C=()>(String);

// #1364
fn foo() {
    convex_shape.set_point(0, &Vector2f { x: 400.0, y: 100.0 });
    convex_shape.set_point(1, &Vector2f { x: 500.0, y: 70.0 });
    convex_shape.set_point(2, &Vector2f { x: 450.0, y: 100.0 });
    convex_shape.set_point(3, &Vector2f { x: 580.0, y: 150.0 });
}

fn main() {
    let x = Bar;

    // Comment
    let y = Foo {a: x };

    Foo { a: foo() /* comment*/, /* comment*/ b: bar(), ..something };

    Fooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo { a: f(), b: b(), };

    Foooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo { a: f(), b: b(), };

    Foooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo {
        // Comment
        a: foo(), // Comment
        // Comment
        b: bar(), // Comment
    };

    Foo { a:Bar,
          b:f() };

    Quux { x: if cond { bar(); }, y: baz() };

    A { 
    // Lorem ipsum dolor sit amet, consectetur adipiscing elit. Donec a diam lectus. Sed sit amet ipsum mauris. Maecenas congue ligula ac quam viverra nec consectetur ante hendrerit. Donec et mollis dolor.
    first: item(),
        // Praesent et diam eget libero egestas mattis sit amet vitae augue.
        // Nam tincidunt congue enim, ut porta lorem lacinia consectetur.
        second: Item
    };

    Some(Data::MethodCallData(MethodCallData {
        span: sub_span.unwrap(),
        scope: self.enclosing_scope(id),
        ref_id: def_id,
        decl_id: Some(decl_id),
    }));

    Diagram { /*                 o        This graph demonstrates how                  
               *                / \       significant whitespace is           
               *               o   o      preserved.  
               *              /|\   \
               *             o o o   o */
              graph: G, }
}

fn matcher() {
    TagTerminatedByteMatcher {
        matcher: ByteMatcher {
        pattern: b"<HTML",
        mask: b"\xFF\xDF\xDF\xDF\xDF\xFF",
    },
    };
}

fn issue177() {
    struct Foo<T> { memb: T }
    let foo = Foo::<i64> { memb: 10 };
}

fn issue201() {
    let s = S{a:0, ..  b};
}

fn issue201_2() {
    let s = S{a: S2{    .. c}, ..  b};
}

fn issue278() {
    let s = S {
        a: 0,
        //       
        b: 0,
    };
    let s1 = S {
        a: 0,
        // foo
        //      
        // bar
        b: 0,
    };
}

fn struct_exprs() {
    Foo
    { a :  1, b:f( 2)};
    Foo{a:1,b:f(2),..g(3)};
    LoooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooongStruct { ..base };
    IntrinsicISizesContribution { content_intrinsic_sizes: IntrinsicISizes { minimum_inline_size: 0, }, };
}

fn issue123() {
    Foo { a: b, c: d, e: f };

    Foo { a: bb, c: dd, e: ff };

    Foo { a: ddddddddddddddddddddd, b: cccccccccccccccccccccccccccccccccccccc };
}

fn issue491() {
    Foo {
        guard: None,
        arm: 0, // Comment
    };

    Foo {
        arm: 0, // Comment
    };

    Foo { a: aaaaaaaaaa, b: bbbbbbbb, c: cccccccccc, d: dddddddddd, /* a comment */
      e: eeeeeeeee };
}

fn issue698() {
    Record {
        ffffffffffffffffffffffffffields: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa,
    };
    Record {
        ffffffffffffffffffffffffffields: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa,
    }
}

fn issue835() {
    MyStruct {};
    MyStruct { /* a comment */ };
    MyStruct {
        // Another comment
    };
    MyStruct {}
}

fn field_init_shorthand() {
    MyStruct { x, y, z };
    MyStruct { x, y, z, .. base };
    Foo { aaaaaaaaaa, bbbbbbbb, cccccccccc, dddddddddd, /* a comment */
        eeeeeeeee };
    Record { ffffffffffffffffffffffffffieldsaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa };
}
