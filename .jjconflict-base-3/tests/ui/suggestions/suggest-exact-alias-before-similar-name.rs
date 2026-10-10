struct Reader;
//~^ NOTE method `read_exact_buf` not found for this struct

impl Reader {
    fn read_exact(&self) {}

    #[doc(alias("read_exact_buf"))]
    fn read_buf_exact(&self) {}
}

fn main() {
    Reader.read_exact_buf();
    //~^ ERROR no method named `read_exact_buf` found for struct `Reader` in the current scope
    //~^^ HELP there is a method `read_buf_exact` with a similar name
}
