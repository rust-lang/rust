// rustfmt-config: visual_struct_lits.toml

// Struct literal expressions.

fn main() {
    let x = Bar;

    // Comment
    let y = Foo { a: x };

    Foo { a: foo(), // comment
          // comment
          b: bar(),
          ..something };

    Fooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo { a: foo(),
                                                                               b: bar(), };

    Foooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo { // Comment
                                                                                        a: foo(), /* C
                                                                                                   * o
                                                                                                   * m
                                                                                                   * m
                                                                                                   * e
                                                                                                   * n
                                                                                                   * t */
                                                                                        // Comment
                                                                                        b: bar(), /* C
                                                                                                   * o
                                                                                                   * m
                                                                                                   * m
                                                                                                   * e
                                                                                                   * n
                                                                                                   * t */ };

    Foo { a: Bar, b: foo() };

    A { // Lorem ipsum dolor sit amet, consectetur adipiscing elit. Donec a diam lectus. Sed sit
        // amet ipsum mauris. Maecenas congue ligula ac quam viverra nec consectetur ante
        // hendrerit. Donec et mollis dolor.
        first: item(),
        // Praesent et diam eget libero egestas mattis sit amet vitae augue.
        // Nam tincidunt congue enim, ut porta lorem lacinia consectetur.
        second: Item, };

    Diagram { //                 o        This graph demonstrates how
              //                / \       significant whitespace is
              //               o   o      preserved.
              //              /|\   \
              //             o o o   o
              graph: G, }
}
