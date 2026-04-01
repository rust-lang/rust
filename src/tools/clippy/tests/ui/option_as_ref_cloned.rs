#![warn(clippy::option_as_ref_cloned)]
#![allow(clippy::clone_on_copy)]

fn main() {
    let mut x = Some(String::new());

    let _: Option<String> = x.as_ref().cloned();
    //~^ option_as_ref_cloned
    let _: Option<String> = x.as_mut().cloned();
    //~^ option_as_ref_cloned

    let y = x.as_ref();
    let _: Option<&String> = y.as_ref().cloned();
    //~^ option_as_ref_cloned

    macro_rules! cloned_recv {
        () => {
            x.as_ref()
        };
    }

    // Don't lint when part of the expression is from a macro
    let _: Option<String> = cloned_recv!().cloned();
}
