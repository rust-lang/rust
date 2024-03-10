//@ normalize-stderr-test: "`: .*\(os error" -> "`: $$FILE_NOT_FOUND_MSG (os error"

fn main() {
    let _ = include_str!("include-macros/file.txt");            //~ ERROR couldn't read
                                                                //~^HELP different directory
    let _ = include_str!("hello.rs");                           //~ ERROR couldn't read
                                                                //~^HELP different directory
    let _ = include_bytes!("../../data.bin");                   //~ ERROR couldn't read
                                                                //~^HELP different directory
    let _ = include_str!("tests/ui/include-macros/file.txt");   //~ ERROR couldn't read
                                                                //~^HELP different directory
}
