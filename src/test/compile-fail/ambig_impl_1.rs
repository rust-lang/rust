impl methods1 for uint { fn me() -> uint { self } } //! NOTE candidate #1 is methods1::me
impl methods2 for uint { fn me() -> uint { self } } //! NOTE candidate #2 is methods2::me
fn main() { 1u.me(); } //! ERROR multiple applicable methods in scope
