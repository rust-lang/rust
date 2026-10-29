#![feature(staged_api)]
#![feature(inherent_associated_types)]
#![stable(feature = "main", since = "1.0.0")]

#[stable(feature = "main", since = "1.0.0")]
pub struct Owner;

impl Owner {
    #[unstable(feature = "data", issue = "none")]
    pub type Data = ();
}
