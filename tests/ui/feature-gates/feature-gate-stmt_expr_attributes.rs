const X: i32 = #[allow(dead_code)] 8;
//~^ ERROR attributes on expressions are experimental

const Y: i32 =
    /// foo
//~^ ERROR attributes on expressions are experimental
    8;

const Z: i32 = {
    //! foo
//~^ ERROR attributes on expressions are experimental
    8
};

fn main() {}
