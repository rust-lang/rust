//@ normalize-stderr: "`: .*" -> "`: $$FILE_NOT_FOUND_MSG"

fn main() {
    let _ = include_str!("include-macros/file.txt");            //~ ERROR couldn't read
                                                                //~^HELP different directory
    let _ = include_bytes!("../../data.bin");                   //~ ERROR couldn't read
                                                                //~^HELP different directory
    let _ = include_str!("tests/ui/include-macros/file.txt");   //~ ERROR couldn't read
                                                                //~^HELP different directory
}
