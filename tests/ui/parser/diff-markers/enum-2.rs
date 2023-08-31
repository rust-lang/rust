enum E {
    Foo {
<<<<<<< HEAD //~ ERROR encountered git conflict marker
        x: u8,
|||||||
        z: (),
=======
        y: i8,
>>>>>>> branch
    }
}
