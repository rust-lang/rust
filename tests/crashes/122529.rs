//@ known-bug: #122529
pub trait Archive {
    type Archived;
}

impl<'a> Archive for <&'a [u8] as Archive>::Archived {
    type Archived = ();
}
