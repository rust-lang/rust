//@no-rustfix

#![warn(clippy::too_long_first_doc_paragraph)]

pub mod foo {

    // in foo.rs
    //! A very short summary.
    //~^ too_long_first_doc_paragraph
    //! A much longer explanation that goes into a lot more detail about
    //! how the thing works, possibly with doclinks and so one,
    //! and probably spanning a many rows. Blablabla, it needs to be over
    //! 200 characters so I needed to write something longeeeeeeer.
}

/// Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nunc turpis nunc, lacinia
//~^ too_long_first_doc_paragraph
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
//~^ too_long_first_doc_paragraph
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

#[rustfmt::skip]
/// Some function. This doc-string paragraph is too long. Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.
//~^ too_long_first_doc_paragraph
///
/// Here's a second paragraph. It would be preferable to put the details here.
pub fn issue_14274() {}

fn main() {
    // test code goes here
}
