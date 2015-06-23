// Test tuple litterals

fn foo() {
    let a = (a, a, a, a, a);
    let aaaaaaaaaaaaaaaa = (aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa, aaaaaaaaaaaaaa, aaaaaaaaaaaaaa);
    let aaaaaaaaaaaaaaaaaaaaaa = (aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa,
                                  aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa,
                                  aaaaaaaaaaaaaaaaaaaaaaaaa,
                                  aaaa);
    let a = (a,);

    let b = (// This is a comment
             b, // Comment
             b /* Trailing comment */);
}
