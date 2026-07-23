#![feature(edition_redirect)]

mod source {
    pub struct Old;
    pub struct Current;
}

#[rustc_edition_redirect(before = "2024", target(source::Old))]
//~^ ERROR `#[rustc_edition_redirect]` can only be applied to a single import
pub use source::{Current};

#[rustc_edition_redirect(before = "2024", target(source::Old))]
//~^ ERROR `#[rustc_edition_redirect]` can only be applied to a single import
pub use source::*;

mod private {
    pub(crate) struct Old {
        _private: (),
    }
}

#[rustc_edition_redirect(before = "2024", target(private::Old))]
//~^ ERROR edition redirect target `private::Old` is less visible than the redirected item
pub struct Public {
    _private: (),
}

#[rustc_edition_redirect(before = "2024", target(private::Missing))]
//~^ ERROR cannot find `Missing` in `private`
pub struct Unresolved {
    _private: (),
}

struct DuplicateTarget;

#[rustc_edition_redirect(before = "2024", target(DuplicateTarget))]
#[rustc_edition_redirect(before = "2024", target(DuplicateTarget))]
//~^ ERROR multiple edition redirects before edition 2024
struct Duplicate;

mod ambiguous_target {
    mod first {
        pub struct Old {
            _private: (),
        }
    }

    mod second {
        pub struct Old {
            _private: (),
        }
    }

    use self::first::*;
    use self::second::*;

    #[rustc_edition_redirect(before = "2024", target(Old))]
    //~^ ERROR `Old` is ambiguous
    pub struct Ambiguous {
        _private: (),
    }
}

fn main() {}
