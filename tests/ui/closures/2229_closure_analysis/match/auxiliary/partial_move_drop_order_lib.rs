pub struct LoudDrop(pub &'static str);
impl Drop for LoudDrop {
    fn drop(&mut self) {
        println!("dropping {}", self.0);
    }
}

#[non_exhaustive]
pub enum ExtNonExhaustive {
    One(i32, LoudDrop),
}
