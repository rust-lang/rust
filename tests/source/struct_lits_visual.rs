// rustfmt-wrap_comments: true
// rustfmt-struct_lit_style: Visual

// Struct literal expressions.

fn main() {
    let x = Bar;

    // Comment
    let y = Foo {a: x };

    Foo { a: foo() /* comment*/, /* comment*/ b: bar(), ..something };

    Fooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo { a: f(), b: b(), };

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

    Diagram { /*                 o        This graph demonstrates how                  
               *                / \       significant whitespace is           
               *               o   o      preserved.  
               *              /|\   \
               *             o o o   o */
              graph: G, }
}
