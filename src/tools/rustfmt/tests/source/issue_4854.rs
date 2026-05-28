struct Struct {
    // Multiline comment
    // should be formatted
    // properly.
}

struct Struct2 {
    // This formatting
// Should be changed
}

struct Struct3(
    // This
    // is
    // correct
);

struct Struct4(
    // This
// is
// not
// correct
);

struct Struct5 {
    /*
    Comment block
    with many lines.
    */
}

struct Struct6(
    /*
    Comment block
    with many lines.
    */
);

struct Struct7 {
    /*
Invalid
format
*/
}

struct Struct8(
    /*
Invalid
format
*/
);

struct Struct9 { /* bar */ }

struct Struct10 { /* bar
baz
*/ }

mod module {
    struct Struct {
        // Multiline comment
        // should be formatted
        // properly.
    }

    struct Struct2 {
        // This formatting
// Should be changed
    }

    struct Struct3(
        // This
        // is
        // correct
    );

    struct Struct4(
        // This
    // is
    // not
// correct
    );

    struct Struct5 {
        /*
        Comment block
        with many lines.
         */
    }

    struct Struct6(
        /*
        Comment block
        with many lines.
        */
    );

    struct Struct7 {
        /*
Invalid
format
*/
    }

    struct Struct8(
        /*
Invalid
format
*/
    );

    struct Struct9 { /* bar */ }
}
