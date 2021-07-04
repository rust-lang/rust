// run-pass

#![feature(try_trait_v2)]
#![feature(const_trait_impl)]
#![feature(const_try)]
#![feature(const_convert)]

fn main() {
    const fn foo() -> Result<bool, ()> {
        Err(())?;
        Ok(true)
    }

    const FOO: Result<bool, ()> = foo();
    assert_eq!(Err(()), FOO);
}
