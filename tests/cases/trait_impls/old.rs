pub struct Def;

pub trait Abc { }

impl<T> Abc for Option<T> { }

impl Abc for Def { }

impl<T> Abc for Box<T> { }
