// rustfmt should not remove doc comments or comments inside attributes.

/**
This function has a block doc comment.
 */
fn test_function() {}

#[foo /* do not remove this! */]
fn foo() {}
