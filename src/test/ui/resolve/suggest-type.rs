use std::ffi::CString;

mod foo {
    fn bar() {}
}

fn main() {
    let _ = Cstring::new("hello").unwrap();
    //~^ ERROR failed to resolve: use of undeclared type or module `Cstring`
    //~| HELP a struct with a similar name exists
    //~| SUGGESTION CString

    let _ = foO::bar();
    //~^ ERROR failed to resolve: use of undeclared type or module `foO`
    //~| HELP a module with a similar name exists
    //~| SUGGESTION foo
}
