trait ZstAssert: Sized {
    const TYPE_NAME: &str = ""; //~ ERROR missing lifetime specifier
}

fn main() {}
