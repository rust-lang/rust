// rustfmt-item_brace_style: AlwaysNextLine

mod M {
    enum A
    {
        A,
    }

    struct B
    {
        b: i32,
    }

    // For empty enums and structs, the brace remains on the same line.
    enum C {}

    struct D {}
}
