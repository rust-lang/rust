pub struct Def;

pub trait Abc { }

impl<T> Abc for Option<T> { }

impl Abc for Def { }

impl<T: Clone> Abc for Box<T> { }

impl Abc for Box<Def> { }
