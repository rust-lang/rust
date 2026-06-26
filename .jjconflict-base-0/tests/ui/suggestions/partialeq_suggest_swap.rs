struct T(i32);

impl PartialEq<i32> for T {
    fn eq(&self, other: &i32) -> bool {
        &self.0 == other
    }
}

fn main() {
    4i32 == T(4); //~ ERROR mismatched types [E0308]
}
