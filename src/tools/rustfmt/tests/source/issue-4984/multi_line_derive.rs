#[derive(
/* ---------- Some really important comment that just had to go inside the derive --------- */
Debug, Clone, Eq, PartialEq,
)]
struct Foo {
    a: i32,
    b: T,
}

#[derive(
/*
    Some really important comment that just had to go inside the derive.
    Also had to be put over multiple lines
*/
Debug, Clone, Eq, PartialEq,
)]
struct Bar {
    a: i32,
    b: T,
}
