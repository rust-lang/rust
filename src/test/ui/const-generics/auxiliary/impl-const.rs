#![cfg_attr(full, feature(const_generics))]
#![cfg_attr(full, allow(incomplete_features))]

pub struct Num<const N: usize>;

// Braces around const expression causes crash
impl Num<{5}> {
    pub fn five(&self) {
    }
}
