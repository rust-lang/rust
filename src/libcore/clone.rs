/**
Clonable types are copied with the clone method
*/
pub trait Clone {
    fn clone(&self) -> self;
}

impl (): Clone {
    fn clone(&self) -> () { () }
}
