// rustfmt-brace_style: AlwaysNextLine
mod x
{
    struct X(i8);

    impl Y for X
    {
        fn y(self) -> ()
        {
            println!("ok");
        }
    }
}
