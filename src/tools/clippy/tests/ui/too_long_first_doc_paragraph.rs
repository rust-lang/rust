//@no-rustfix

#![warn(clippy::too_long_first_doc_paragraph)]

pub mod foo {

    // in foo.rs
    //! A very short summary.
    //! A much longer explanation that goes into a lot more detail about
    //! how the thing works, possibly with doclinks and so one,
    //! and probably spanning a many rows. Blablabla, it needs to be over
    //! 200 characters so I needed to write something longeeeeeeer.
}

/// Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nunc turpis nunc, lacinia
/// a dolor in, pellentesque aliquet enim. Cras nec maximus sem. Mauris arcu libero,
/// gravida non lacinia at, rhoncus eu lacus.
pub struct Bar;

// Should not warn! (not an item visible on mod page)
/// Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nunc turpis nunc, lacinia
/// a dolor in, pellentesque aliquet enim. Cras nec maximus sem. Mauris arcu libero,
/// gravida non lacinia at, rhoncus eu lacus.
impl Bar {}

// Should not warn! (less than 80 characters)
/// Lorem ipsum dolor sit amet, consectetur adipiscing elit.
///
/// Nunc turpis nunc, lacinia
/// a dolor in, pellentesque aliquet enim. Cras nec maximus sem. Mauris arcu libero,
/// gravida non lacinia at, rhoncus eu lacus.
pub enum Enum {
    A,
}

/// Lorem
/// ipsum dolor sit amet, consectetur adipiscing elit. Nunc turpis nunc, lacinia
/// a dolor in, pellentesque aliquet enim. Cras nec maximus sem. Mauris arcu libero,
/// gravida non lacinia at, rhoncus eu lacus.
pub union Union {
    a: u8,
    b: u8,
}

// Should not warn! (title)
/// # bla
/// Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nunc turpis nunc, lacinia
/// a dolor in, pellentesque aliquet enim. Cras nec maximus sem. Mauris arcu libero,
/// gravida non lacinia at, rhoncus eu lacus.
pub union Union2 {
    a: u8,
    b: u8,
}

// Should not warn! (not public)
/// Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nunc turpis nunc, lacinia
/// a dolor in, pellentesque aliquet enim. Cras nec maximus sem. Mauris arcu libero,
/// gravida non lacinia at, rhoncus eu lacus.
fn f() {}

fn main() {
    // test code goes here
}
