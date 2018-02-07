// rustfmt-normalize_comments: true
enum E {
    A, //* I am not a block comment (caused panic)
    B,
}
