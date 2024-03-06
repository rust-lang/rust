//@ check-pass
pub struct S(u8);

impl S {
    pub fn get<const A: u8>(&self) -> &u8 {
        &self.0
    }
}

fn main() {
    const A: u8 = 5;
    let s = S(0);

    s.get::<A>();
}
