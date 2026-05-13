//! Regression test for <https://github.com/rust-lang/rust/issues/156200>

//! Fixes misleading "return statement outside of function body" message emitted
//! when `?` was used in a `const`/`static` initializer, even if that initializer
//! was within a function body

const fn foo() -> Result<(), ()> { Ok(()) }

fn main() -> Result<(), ()> {
    const A: () = foo()?; //~ ERROR the `?` operator cannot be used to return from inside a `const` initializer
    static B: () = foo()?; //~ ERROR the `?` operator cannot be used to return from inside a `static` initializer
    Ok(())
}
