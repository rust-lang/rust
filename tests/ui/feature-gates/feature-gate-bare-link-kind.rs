#[link(kind = "dylib")] //~ ERROR `#[link]` attribute requires a `name = "string"` argument
extern "C" {
    static FOO: u32;
}

fn main() {}
