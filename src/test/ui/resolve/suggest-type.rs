use std::ffi::CString;

mod foo {
    use std::collections::HashMap;

    fn bar() {
        let _ = HashNap::new();
        //~^ ERROR failed to resolve: use of undeclared type or module `HashNap`
        //~| HELP a struct with a similar name exists
        //~| SUGGESTION HashMap
    }
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
