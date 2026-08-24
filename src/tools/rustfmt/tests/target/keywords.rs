pub // a
macro // b
hi(
    // c
) {
    // d
}

macro_rules! // a
my_macro {
    () => {};
}

// == comments don't get reformatted ==
macro_rules!// a
  // b
    // c
 // d
my_macro {
    () => {};
}

macro_rules! /* a block comment */
my_macro {
    () => {};
}
