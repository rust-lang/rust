//@ ignore-backends: gcc
//@ run-pass

#![feature(const_trait_impl)]
#![feature(const_try)]

fn main() {
    const fn result() -> Result<bool, ()> {
        Err(())?;
        Ok(true)
    }

    const FOO: Result<bool, ()> = result();
    assert_eq!(Err(()), FOO);

    const fn option() -> Option<()> {
        None?;
        Some(())
    }
    const BAR: Option<()> = option();
    assert_eq!(None, BAR);
}
