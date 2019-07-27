// run-pass
#![allow(dead_code)]
#![allow(unused_variables)]
// pretty-expanded FIXME #23616

enum StdioContainer {
    CreatePipe(bool)
}

struct Test<'a> {
    args: &'a [String],
    io: &'a [StdioContainer]
}

pub fn main() {
    let test = Test {
        args: &[],
        io: &[StdioContainer::CreatePipe(true)]
    };
}
