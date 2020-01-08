#![feature(optin_builtin_traits)]

pub trait ForeignTrait { }

impl ForeignTrait for u32 { }
impl !ForeignTrait for String {}
