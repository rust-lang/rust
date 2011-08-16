// Issue #679
// Testing that comments are correctly interleaved
// pp-exact:vec-comments.pp
fn main() {
    let v1 = ~[
        // Comment
        0,
        // Comment
        1,
        // Comment
        2
    ];
    let v2 = ~[
        0, // Comment
        1, // Comment
        2  // Comment
    ];
    let v3 = ~[
        /* Comment */
        0,
        /* Comment */
        1,
        /* Comment */
        2
    ];
    let v4 = ~[
        0, /* Comment */
        1, /* Comment */
        2  /* Comment */
    ];
}
