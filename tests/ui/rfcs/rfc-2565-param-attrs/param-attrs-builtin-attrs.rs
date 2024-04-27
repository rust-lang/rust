extern "C" {
    fn ffi(
        /// Foo
        //~^ ERROR documentation comments cannot be applied to function
        #[test] a: i32,
        //~^ ERROR expected non-macro attribute, found attribute macro
        /// Bar
        //~^ ERROR documentation comments cannot be applied to function
        #[must_use]
        //~^ ERROR allow, cfg, cfg_attr, deny, expect, forbid, and warn are the only allowed built-in attributes in function parameters
        /// Baz
        //~^ ERROR documentation comments cannot be applied to function
        #[no_mangle] b: i32,
        //~^ ERROR allow, cfg, cfg_attr, deny, expect, forbid, and warn are the only allowed built-in attributes in function parameters
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
    //~^ ERROR allow, cfg, cfg_attr, deny, expect, forbid, and warn are the only allowed built-in attributes in function parameters
    /// Baz
    //~^ ERROR documentation comments cannot be applied to function
    #[no_mangle] b: i32,
    //~^ ERROR allow, cfg, cfg_attr, deny, expect, forbid, and warn are the only allowed built-in attributes in function parameters
);

pub fn foo(
    /// Foo
    //~^ ERROR documentation comments cannot be applied to function
    #[test] a: u32,
    //~^ ERROR expected non-macro attribute, found attribute macro
    /// Bar
    //~^ ERROR documentation comments cannot be applied to function
    #[must_use]
    //~^ ERROR allow, cfg, cfg_attr, deny, expect, forbid, and warn are the only allowed built-in attributes in function parameters
    /// Baz
    //~^ ERROR documentation comments cannot be applied to function
    #[no_mangle] b: i32,
    //~^ ERROR allow, cfg, cfg_attr, deny, expect, forbid, and warn are the only allowed built-in attributes in function parameters
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
        //~^ ERROR allow, cfg, cfg_attr, deny, expect, forbid, and warn are the only allowed built-in attributes in function parameters
        /// Qux
        //~^ ERROR documentation comments cannot be applied to function
        #[no_mangle] b: i32,
        //~^ ERROR allow, cfg, cfg_attr, deny, expect, forbid, and warn are the only allowed built-in attributes in function parameters
    ) {}

    fn issue_64682_associated_fn(
        /// Foo
        //~^ ERROR documentation comments cannot be applied to function
        #[test] a: i32,
        //~^ ERROR expected non-macro attribute, found attribute macro
        /// Baz
        //~^ ERROR documentation comments cannot be applied to function
        #[must_use]
        //~^ ERROR allow, cfg, cfg_attr, deny, expect, forbid, and warn are the only allowed built-in attributes in function parameters
        /// Qux
        //~^ ERROR documentation comments cannot be applied to function
        #[no_mangle] b: i32,
        //~^ ERROR allow, cfg, cfg_attr, deny, expect, forbid, and warn are the only allowed built-in attributes in function parameters
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
        //~^ ERROR allow, cfg, cfg_attr, deny, expect, forbid, and warn are the only allowed built-in attributes in function parameters
        /// Qux
        //~^ ERROR documentation comments cannot be applied to function
        #[no_mangle] b: i32,
        //~^ ERROR allow, cfg, cfg_attr, deny, expect, forbid, and warn are the only allowed built-in attributes in function parameters
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
        //~^ ERROR allow, cfg, cfg_attr, deny, expect, forbid, and warn are the only allowed built-in attributes in function parameters
        /// Qux
        //~^ ERROR documentation comments cannot be applied to function
        #[no_mangle] b: i32,
        //~^ ERROR allow, cfg, cfg_attr, deny, expect, forbid, and warn are the only allowed built-in attributes in function parameters
    ) {}

    fn issue_64682_associated_fn(
        /// Foo
        //~^ ERROR documentation comments cannot be applied to function
        #[test] a: i32,
        //~^ ERROR expected non-macro attribute, found attribute macro
        /// Baz
        //~^ ERROR documentation comments cannot be applied to function
        #[must_use]
        //~^ ERROR allow, cfg, cfg_attr, deny, expect, forbid, and warn are the only allowed built-in attributes in function parameters
        /// Qux
        //~^ ERROR documentation comments cannot be applied to function
        #[no_mangle] b: i32,
        //~^ ERROR allow, cfg, cfg_attr, deny, expect, forbid, and warn are the only allowed built-in attributes in function parameters
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
        //~^ ERROR allow, cfg, cfg_attr, deny, expect, forbid, and warn are the only allowed built-in attributes in function parameters
        /// Qux
        //~^ ERROR documentation comments cannot be applied to function
        #[no_mangle] b: i32,
        //~^ ERROR allow, cfg, cfg_attr, deny, expect, forbid, and warn are the only allowed built-in attributes in function parameters
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
        //~^ ERROR allow, cfg, cfg_attr, deny, expect, forbid, and warn are the only allowed built-in attributes in function parameters
        /// Baz
        //~^ ERROR documentation comments cannot be applied to function
        #[no_mangle] b: i32
        //~^ ERROR allow, cfg, cfg_attr, deny, expect, forbid, and warn are the only allowed built-in attributes in function parameters
    | {};
}
