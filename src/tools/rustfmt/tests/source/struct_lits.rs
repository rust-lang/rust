// rustfmt-normalize_comments: true
// rustfmt-wrap_comments: true
// Struct literal expressions.

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
