mod some_module {
    compile_error!("Error in a module"); //~ ERROR: Error in a module

    fn abc() {
        let _: self::SomeType = self::Hello::new();
        let _: SomeType = Hello::new();
    }
}

mod another_module {}

fn main() {
    // these errors are suppressed because of the compile_error! macro

    let _ = some_module::some_function();
    let _: some_module::SomeType = some_module::Hello::new();

    // these errors are not suppressed

    let _ = another_module::some_function();
    //~^ ERROR: cannot find function `some_function` in module `another_module`
    let _: another_module::SomeType = another_module::Hello::new();
    //~^ ERROR: cannot find type `SomeType` in module `another_module`
    //~^^ ERROR: failed to resolve: could not find `Hello` in `another_module`
}
