pub mod some_module {
    compile_error!("Error in a module"); //~ ERROR: Error in a module

    fn abc() -> Hello {
        let _: self::SomeType = self::Hello::new();
        let _: SomeType = Hello::new();
    }

    mod inner_module {
        use super::Hello;
        use crate::another_module::NotExist;  //~ ERROR: unresolved import `crate::another_module::NotExist`
        use crate::some_module::World;
        struct Foo {
            bar: crate::some_module::Xyz,
            error: self::MissingType, //~ ERROR: cannot find type `MissingType` in module `self`
        }
    }
}

pub mod another_module {
    use crate::some_module::NotExist;
    fn error_in_this_function() {
        compile_error!("Error in a function"); //~ ERROR: Error in a function
    }
}

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
