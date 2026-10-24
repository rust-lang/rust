fn f() -> Box<
    dyn FnMut() -> Thing<
        WithType = LongItemName,
        Error = LONGLONGLONGLONGLONGONGEvenLongerErrorNameLongerLonger,
    >,
> {
}
