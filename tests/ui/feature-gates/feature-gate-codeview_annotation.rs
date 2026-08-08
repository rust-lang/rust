fn main() {
    std::hint::codeview_annotation!("string1", "string2", "string3"); //~ ERROR use of unstable library feature `codeview_annotation`
    //~| ERROR use of unstable library feature `codeview_annotation`
}
