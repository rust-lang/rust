// Tests that "loop unsafe" isn't misparsed.

fn main() {
    loop unsafe {
        io::println("Hello world!");
        return ();
    }
}

