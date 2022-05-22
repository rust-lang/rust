extern "C" {
    fn ffi(
        /// Foo
        //~^ ERROR documentation comments cannot be applied to function
        #[test] a: i32,
        //~^ ERROR expected non-macro attribute, found attribute macro
        /// Bar
        //~^ ERROR documentation comments cannot be applied to function
        #[must_use]
        //~^ ERROR allow, cfg, cfg_attr, deny, forbid, and warn are the only allowed built-in
        /// Baz
        //~^ ERROR documentation comments cannot be applied to function
        #[no_mangle] b: i32,
        //~^ ERROR allow, cfg, cfg_attr, deny, forbid, and warn are the only allowed built-in
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
    //~^ ERROR allow, cfg, cfg_attr, deny, forbid, and warn are the only allowed built-in
    /// Baz
    //~^ ERROR documentation comments cannot be applied to function
    #[no_mangle] b: i32,
    //~^ ERROR allow, cfg, cfg_attr, deny, forbid, and warn are the only allowed built-in
);

pub fn foo(
    /// Foo
    //~^ ERROR documentation comments cannot be applied to function
    #[test] a: u32,
    //~^ ERROR expected non-macro attribute, found attribute macro
    /// Bar
    //~^ ERROR documentation comments cannot be applied to function
    #[must_use]
    //~^ ERROR allow, cfg, cfg_attr, deny, forbid, and warn are the only allowed built-in
    /// Baz
    //~^ ERROR documentation comments cannot be applied to function
    #[no_mangle] b: i32,
    //~^ ERROR allow, cfg, cfg_attr, deny, forbid, and warn are the only allowed built-in
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
        //~^ ERROR allow, cfg, cfg_attr, deny, forbid, and warn are the only allowed built-in
        /// Qux
        //~^ ERROR documentation comments cannot be applied to function
        #[no_mangle] b: i32,
        //~^ ERROR allow, cfg, cfg_attr, deny, forbid, and warn are the only allowed built-in
    ) {}

    fn issue_64682_associated_fn(
        /// Foo
        //~^ ERROR documentation comments cannot be applied to function
        #[test] a: i32,
        //~^ ERROR expected non-macro attribute, found attribute macro
        /// Baz
        //~^ ERROR documentation comments cannot be applied to function
        #[must_use]
        //~^ ERROR allow, cfg, cfg_attr, deny, forbid, and warn are the only allowed built-in
        /// Qux
        //~^ ERROR documentation comments cannot be applied to function
        #[no_mangle] b: i32,
        //~^ ERROR allow, cfg, cfg_attr, deny, forbid, and warn are the only allowed built-in
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
        //~^ ERROR allow, cfg, cfg_attr, deny, forbid, and warn are the only allowed built-in
        /// Qux
        //~^ ERROR documentation comments cannot be applied to function
        #[no_mangle] b: i32,
        //~^ ERROR allow, cfg, cfg_attr, deny, forbid, and warn are the only allowed built-in
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
        //~^ ERROR allow, cfg, cfg_attr, deny, forbid, and warn are the only allowed built-in
        /// Qux
        //~^ ERROR documentation comments cannot be applied to function
        #[no_mangle] b: i32,
        //~^ ERROR allow, cfg, cfg_attr, deny, forbid, and warn are the only allowed built-in
    ) {}

    fn issue_64682_associated_fn(
        /// Foo
        //~^ ERROR documentation comments cannot be applied to function
        #[test] a: i32,
        //~^ ERROR expected non-macro attribute, found attribute macro
        /// Baz
        //~^ ERROR documentation comments cannot be applied to function
        #[must_use]
        //~^ ERROR allow, cfg, cfg_attr, deny, forbid, and warn are the only allowed built-in
        /// Qux
        //~^ ERROR documentation comments cannot be applied to function
        #[no_mangle] b: i32,
        //~^ ERROR allow, cfg, cfg_attr, deny, forbid, and warn are the only allowed built-in
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
        //~^ ERROR allow, cfg, cfg_attr, deny, forbid, and warn are the only allowed built-in
        /// Qux
        //~^ ERROR documentation comments cannot be applied to function
        #[no_mangle] b: i32,
        //~^ ERROR allow, cfg, cfg_attr, deny, forbid, and warn are the only allowed built-in
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
        //~^ ERROR allow, cfg, cfg_attr, deny, forbid, and warn are the only allowed built-in
        /// Baz
        //~^ ERROR documentation comments cannot be applied to function
        #[no_mangle] b: i32
        //~^ ERROR allow, cfg, cfg_attr, deny, forbid, and warn are the only allowed built-in
    | {};
}
