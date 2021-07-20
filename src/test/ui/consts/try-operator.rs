#![feature(const_try)]
#![feature(const_identity_convert)]

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
