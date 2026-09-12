struct Args;

impl std::hint::CodeViewAnnotationArgs for Args { //~ ERROR use of unstable library feature `codeview_annotation`
    const ARGS: &[&str] = &["string1", "string2", "string3"]; //~ ERROR use of unstable library feature `codeview_annotation`
}

fn main() {
    std::hint::codeview_annotation::<Args>(); //~ ERROR use of unstable library feature `codeview_annotation`
}
