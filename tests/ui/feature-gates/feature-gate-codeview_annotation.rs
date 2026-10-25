//@ only-windows

struct Args;

impl std::os::windows::CodeViewAnnotationArgs for Args { //~ ERROR use of unstable library feature `codeview_annotation`
    const ARGS: &[&str] = &["string1", "string2", "string3"]; //~ ERROR use of unstable library feature `codeview_annotation`
}

fn main() {
    std::os::windows::codeview_annotation::<Args>(); //~ ERROR use of unstable library feature `codeview_annotation`
}
