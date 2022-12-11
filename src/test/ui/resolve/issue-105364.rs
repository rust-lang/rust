// compile-flags: --crate-type lib

fn cause_compiler_bug() {
    let content_line = content_line::fail(); //~ ERROR failed to resolve: use of undeclared crate or module `content_line`
}
