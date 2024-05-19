pub trait Tr {
    type VisibleAssoc;
    #[doc(hidden)]
    type HiddenAssoc;

    const VISIBLE_ASSOC: ();
    #[doc(hidden)]
    const HIDDEN_ASSOC: ();
}

pub struct Ty;

impl Tr for Ty {
    type VisibleAssoc = ();
    type HiddenAssoc = ();

    const VISIBLE_ASSOC: () = ();
    const HIDDEN_ASSOC: () = ();
}
