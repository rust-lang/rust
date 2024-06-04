//! This test used to ICE: #123917
//! The reason was that while the AST knows about two fields
//! named `ptr`, only one exists at the layout level, so accessing
//! `_extra_field` would use an oob index
//@ compile-flags: -Zmir-opt-level=5 -Zpolymorphize=on

struct NonNull<T>(*mut T);

struct Token<T> {
    ptr: *mut T,
    ptr: NonNull<T>,
    //~^ ERROR: `ptr` is already declared
    _extra_field: (),
}

fn tokenize<T>(item: *mut T) -> Token<T> {
    Token { ptr: NonNull(item), _extra_field: () }
}

fn main() {}
