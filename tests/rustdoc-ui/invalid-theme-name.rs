//@ compile-flags:--theme {{src-base}}/invalid-theme-name.rs

//~? ERROR invalid argument: "$DIR/invalid-theme-name.rs"
//~? HELP must have a .css extension
