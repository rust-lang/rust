#![feature(const_generics)]

pub struct Num<const N: usize>;

// Braces around const expression causes crash
impl Num<{5}> {
    pub fn five(&self) {
    }
}
