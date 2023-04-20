#![allow(unused)]
#![allow(clippy::identity_op)]
#![allow(clippy::no_effect)]
#![warn(clippy::excessive_width)]

static mut C: u32 = 2u32;

#[rustfmt::skip]
fn main() {
    let x = 2 * unsafe { C };

    {
        {
            // this too, even though it's only 15 characters!
            ();
        }
    }

    {
        {
            {
                println!("this will now emit a warning, how neat!")
            }
        }
    }
}
