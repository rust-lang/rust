#![feature(const_trait_impl)]
#![feature(staged_api)]
#![stable(feature = "rust1", since = "1.0.0")]

#[stable(feature = "rust1", since = "1.0.0")]
pub struct Int(i32);

#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_const_stable(feature = "rust1", since = "1.0.0")]
impl const std::ops::Sub for Int {
    //~^ ERROR trait implementations cannot be const stable yet
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Int(self.0 - rhs.0)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_const_unstable(feature = "const_add", issue = "none")]
impl const std::ops::Add for Int {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Int(self.0 + rhs.0)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_const_stable(feature = "rust1", since = "1.0.0")]
pub const fn const_err() {
    Int(0) + Int(0);
    //~^ ERROR not yet stable as a const fn
    Int(0) - Int(0);
}

#[stable(feature = "rust1", since = "1.0.0")]
pub fn non_const_success() {
    Int(0) + Int(0);
    Int(0) - Int(0);
}

fn main() {}
