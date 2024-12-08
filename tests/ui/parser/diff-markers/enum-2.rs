enum E {
    Foo {
<<<<<<< HEAD //~ ERROR encountered diff marker
        x: u8,
|||||||
        z: (),
=======
        y: i8,
>>>>>>> branch
    }
}
