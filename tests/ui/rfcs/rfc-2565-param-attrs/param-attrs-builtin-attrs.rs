extern "C" {
    fn ffi(
        /// Foo
        //~^ ERROR documentation comments cannot be applied to function
        #[test] a: i32,
        //~^ ERROR expected non-macro attribute, found attribute macro
        /// Bar
        //~^ ERROR documentation comments cannot be applied to function
        #[must_use]
        //~^ ERROR attribute cannot be used on
        /// Baz
        //~^ ERROR documentation comments cannot be applied to function
        #[no_mangle] b: i32,
        //~^ ERROR attribute cannot be used on
    );
}

type FnType = fn(
    /// Foo
    //~^ ERROR documentation comments cannot be applied to function
    #[test] a: u32,
    //~^ ERROR expected non-macro attribute, found attribute macro
    /// Bar
    //~^ ERROR documentation comments cannot be applied to function
    #[must_use]
    //~^ ERROR attribute cannot be used on
    /// Baz
    //~^ ERROR documentation comments cannot be applied to function
    #[no_mangle] b: i32,
    //~^ ERROR attribute cannot be used on
);

pub fn foo(
    /// Foo
    //~^ ERROR documentation comments cannot be applied to function
    #[test] a: u32,
    //~^ ERROR expected non-macro attribute, found attribute macro
    /// Bar
    //~^ ERROR documentation comments cannot be applied to function
    #[must_use]
    //~^ ERROR attribute cannot be used on
    /// Baz
    //~^ ERROR documentation comments cannot be applied to function
    #[no_mangle] b: i32,
    //~^ ERROR attribute cannot be used on
) {}

struct SelfStruct {}
impl SelfStruct {
    fn foo(
        /// Foo
        //~^ ERROR documentation comments cannot be applied to function
        self,
        /// Bar
        //~^ ERROR documentation comments cannot be applied to function
        #[test] a: i32,
        //~^ ERROR expected non-macro attribute, found attribute macro
        /// Baz
        //~^ ERROR documentation comments cannot be applied to function
        #[must_use]
        //~^ ERROR attribute cannot be used on
        /// Qux
        //~^ ERROR documentation comments cannot be applied to function
        #[no_mangle] b: i32,
        //~^ ERROR attribute cannot be used on
    ) {}

    fn issue_64682_associated_fn(
        /// Foo
        //~^ ERROR documentation comments cannot be applied to function
        #[test] a: i32,
        //~^ ERROR expected non-macro attribute, found attribute macro
        /// Baz
        //~^ ERROR documentation comments cannot be applied to function
        #[must_use]
        //~^ ERROR attribute cannot be used on
        /// Qux
        //~^ ERROR documentation comments cannot be applied to function
        #[no_mangle] b: i32,
        //~^ ERROR attribute cannot be used on
    ) {}
}

struct RefStruct {}
impl RefStruct {
    fn foo(
        /// Foo
        //~^ ERROR documentation comments cannot be applied to function
        &self,
        /// Bar
        //~^ ERROR documentation comments cannot be applied to function
        #[test] a: i32,
        //~^ ERROR expected non-macro attribute, found attribute macro
        /// Baz
        //~^ ERROR documentation comments cannot be applied to function
        #[must_use]
        //~^ ERROR attribute cannot be used on
        /// Qux
        //~^ ERROR documentation comments cannot be applied to function
        #[no_mangle] b: i32,
        //~^ ERROR attribute cannot be used on
    ) {}
}
trait RefTrait {
    fn foo(
        /// Foo
        //~^ ERROR documentation comments cannot be applied to function
        &self,
        /// Bar
        //~^ ERROR documentation comments cannot be applied to function
        #[test] a: i32,
        //~^ ERROR expected non-macro attribute, found attribute macro
        /// Baz
        //~^ ERROR documentation comments cannot be applied to function
        #[must_use]
        //~^ ERROR attribute cannot be used on
        /// Qux
        //~^ ERROR documentation comments cannot be applied to function
        #[no_mangle] b: i32,
        //~^ ERROR attribute cannot be used on
    ) {}

    fn issue_64682_associated_fn(
        /// Foo
        //~^ ERROR documentation comments cannot be applied to function
        #[test] a: i32,
        //~^ ERROR expected non-macro attribute, found attribute macro
        /// Baz
        //~^ ERROR documentation comments cannot be applied to function
        #[must_use]
        //~^ ERROR attribute cannot be used on
        /// Qux
        //~^ ERROR documentation comments cannot be applied to function
        #[no_mangle] b: i32,
        //~^ ERROR attribute cannot be used on
    ) {}
}

impl RefTrait for RefStruct {
    fn foo(
        /// Foo
        //~^ ERROR documentation comments cannot be applied to function
        &self,
        /// Bar
        //~^ ERROR documentation comments cannot be applied to function
        #[test] a: i32,
        //~^ ERROR expected non-macro attribute, found attribute macro
        /// Baz
        //~^ ERROR documentation comments cannot be applied to function
        #[must_use]
        //~^ ERROR attribute cannot be used on
        /// Qux
        //~^ ERROR documentation comments cannot be applied to function
        #[no_mangle] b: i32,
        //~^ ERROR attribute cannot be used on
    ) {}
}

fn main() {
    let _ = |
        /// Foo
        //~^ ERROR documentation comments cannot be applied to function
        #[test] a: u32,
        //~^ ERROR expected non-macro attribute, found attribute macro
        /// Bar
        //~^ ERROR documentation comments cannot be applied to function
        #[must_use]
        //~^ ERROR attribute cannot be used on
        /// Baz
        //~^ ERROR documentation comments cannot be applied to function
        #[no_mangle] b: i32
        //~^ ERROR attribute cannot be used on
    | {};
}
