extern "C" {
    fn ffi(
        /// Foo
        //~^ ERROR unused doc comment
        #[test] a: i32,
        //~^ ERROR expected non-macro attribute, found attribute macro
        /// Bar
        //~^ ERROR unused doc comment
        #[must_use]
        //~^ ERROR cannot be used on function params
        /// Baz
        #[no_mangle] b: i32,
        //~^ ERROR cannot be used on function params
    );
}

type FnType = fn(
    /// Foo
    //~^ ERROR unused doc comment
    #[test] a: u32,
    //~^ ERROR expected non-macro attribute, found attribute macro
    /// Bar
    //~^ ERROR unused doc comment
    #[must_use]
    //~^ ERROR cannot be used on function params
    /// Baz
    #[no_mangle] b: i32,
    //~^ ERROR cannot be used on function params
);

pub fn foo(
    /// Foo
    //~^ ERROR unused doc comment
    #[test] a: u32,
    //~^ ERROR expected non-macro attribute, found attribute macro
    /// Bar
    //~^ ERROR unused doc comment
    #[must_use]
    //~^ ERROR attribute cannot be used on
    /// Baz
    #[no_mangle] b: i32,
    //~^ ERROR attribute cannot be used on
) {}

struct SelfStruct {}
impl SelfStruct {
    fn foo(
        /// Foo
        //~^ ERROR unused doc comment
        self,
        /// Bar
        //~^ ERROR unused doc comment
        #[test] a: i32,
        //~^ ERROR expected non-macro attribute, found attribute macro
        /// Baz
        //~^ ERROR unused doc comment
        #[must_use]
        //~^ ERROR attribute cannot be used on
        /// Qux
        #[no_mangle] b: i32,
        //~^ ERROR attribute cannot be used on
    ) {}

    fn issue_64682_associated_fn(
        /// Foo
        //~^ ERROR unused doc comment
        #[test] a: i32,
        //~^ ERROR expected non-macro attribute, found attribute macro
        /// Baz
        //~^ ERROR unused doc comment
        #[must_use]
        //~^ ERROR attribute cannot be used on
        /// Qux
        #[no_mangle] b: i32,
        //~^ ERROR attribute cannot be used on
    ) {}
}

struct RefStruct {}
impl RefStruct {
    fn foo(
        /// Foo
        //~^ ERROR unused doc comment
        &self,
        /// Bar
        //~^ ERROR unused doc comment
        #[test] a: i32,
        //~^ ERROR expected non-macro attribute, found attribute macro
        /// Baz
        //~^ ERROR unused doc comment
        #[must_use]
        //~^ ERROR attribute cannot be used on
        /// Qux
        #[no_mangle] b: i32,
        //~^ ERROR attribute cannot be used on
    ) {}
}
trait RefTrait {
    fn foo(
        /// Foo
        //~^ ERROR unused doc comment
        &self,
        /// Bar
        //~^ ERROR unused doc comment
        #[test] a: i32,
        //~^ ERROR expected non-macro attribute, found attribute macro
        /// Baz
        //~^ ERROR unused doc comment
        #[must_use]
        //~^ ERROR attribute cannot be used on
        /// Qux
        #[no_mangle] b: i32,
        //~^ ERROR attribute cannot be used on
    ) {}

    fn issue_64682_associated_fn(
        /// Foo
        //~^ ERROR unused doc comment
        #[test] a: i32,
        //~^ ERROR expected non-macro attribute, found attribute macro
        /// Baz
        //~^ ERROR unused doc comment
        #[must_use]
        //~^ ERROR attribute cannot be used on
        /// Qux
        #[no_mangle] b: i32,
        //~^ ERROR attribute cannot be used on
    ) {}
}

impl RefTrait for RefStruct {
    fn foo(
        /// Foo
        //~^ ERROR unused doc comment
        &self,
        /// Bar
        //~^ ERROR unused doc comment
        #[test] a: i32,
        //~^ ERROR expected non-macro attribute, found attribute macro
        /// Baz
        //~^ ERROR unused doc comment
        #[must_use]
        //~^ ERROR attribute cannot be used on
        /// Qux
        #[no_mangle] b: i32,
        //~^ ERROR attribute cannot be used on
    ) {}
}

fn main() {
    let _ = |
        /// Foo
        //~^ ERROR unused doc comment
        #[test] a: u32,
        //~^ ERROR expected non-macro attribute, found attribute macro
        /// Bar
        //~^ ERROR unused doc comment
        #[must_use]
        //~^ ERROR attribute cannot be used on
        /// Baz
        #[no_mangle] b: i32
        //~^ ERROR attribute cannot be used on
    | {};
}
