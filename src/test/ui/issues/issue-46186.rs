// run-rustfix

pub struct Struct {
    pub a: usize,
};
//~^ ERROR expected item, found `;`

fn main() {}
