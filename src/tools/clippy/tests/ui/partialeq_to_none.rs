// run-rustfix
#![warn(clippy::partialeq_to_none)]

struct Foobar;

impl PartialEq<Option<()>> for Foobar {
    fn eq(&self, _: &Option<()>) -> bool {
        false
    }
}

#[allow(dead_code)]
fn foo(f: Option<u32>) -> &'static str {
    if f != None { "yay" } else { "nay" }
}

fn foobar() -> Option<()> {
    None
}

fn bar() -> Result<(), ()> {
    Ok(())
}

fn optref() -> &'static &'static Option<()> {
    &&None
}

pub fn macro_expansion() {
    macro_rules! foo {
        () => {
            None::<()>
        };
    }

    let _ = foobar() == foo!();
    let _ = foo!() == foobar();
    let _ = foo!() == foo!();
}

fn main() {
    let x = Some(0);

    let _ = x == None;
    let _ = x != None;
    let _ = None == x;
    let _ = None != x;

    if foobar() == None {}

    if bar().ok() != None {}

    let _ = Some(1 + 2) != None;

    let _ = { Some(0) } == None;

    let _ = {
        /*
          This comment runs long
        */
        Some(1)
    } != None;

    // Should not trigger, as `Foobar` is not an `Option` and has no `is_none`
    let _ = Foobar == None;

    let _ = optref() == &&None;
    let _ = &&None != optref();
    let _ = **optref() == None;
    let _ = &None != *optref();

    let x = Box::new(Option::<()>::None);
    let _ = None != *x;
}
