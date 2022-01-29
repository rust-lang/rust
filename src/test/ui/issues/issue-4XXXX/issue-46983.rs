fn foo(x: &u32) -> &'static u32 {
    &*x
    //~^ ERROR `x` has an anonymous lifetime `'_` but it needs to satisfy a `'static` lifetime requirement [E0759]
}

fn main() {}
