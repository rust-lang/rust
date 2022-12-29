enum E {
    Foo {
<<<<<<< HEAD //~ ERROR encountered diff marker
        x: u8,
=======
        x: i8,
>>>>>>> branch
    }
}
