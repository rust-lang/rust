pub struct ZeroLengthThingWithDestructor;
impl Drop for ZeroLengthThingWithDestructor {
    fn drop(&mut self) {}
}
impl ZeroLengthThingWithDestructor {
    pub fn new() -> ZeroLengthThingWithDestructor {
        ZeroLengthThingWithDestructor
    }
}
