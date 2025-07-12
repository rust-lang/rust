fn foo<X>() {}

fn do_something() -> Option<ForgotToImport> {
    //~^ cannot find type `ForgotToImport` in this scope [E0412]
    None
}

fn do_something_T() -> Option<T> {
    //~^ cannot find type `T` in this scope [E0412]
    None
}

fn do_something_Type() -> Option<Type> {
    //~^ cannot find type `Type` in this scope [E0412]
    None
}

fn main() {
    foo(); //~ ERROR type annotations needed
}
