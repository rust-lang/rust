extern "C" {
    #[used] //~ ERROR attribute must be applied to a `static` variable
    static FOO: i32;
}

fn main() {}
