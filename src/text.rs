use std::fmt;
use std::ops;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TextUnit(
    pub(crate) u32
);

impl TextUnit {
    pub fn len_of_char(c: char) -> TextUnit {
        TextUnit(c.len_utf8() as u32)
    }

    pub fn new(val: u32) -> TextUnit {
        TextUnit(val)
    }
}

impl fmt::Debug for TextUnit {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        <Self as fmt::Display>::fmt(self, f)
    }
}

impl fmt::Display for TextUnit {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl From<TextUnit> for u32 {
    fn from(tu: TextUnit) -> u32 {
        tu.0
    }
}

impl ops::Add<TextUnit> for TextUnit {
    type Output = TextUnit;
    fn add(self, rhs: TextUnit) -> TextUnit {
        TextUnit(self.0 + rhs.0)
    }
}

impl ops::AddAssign<TextUnit> for TextUnit {
    fn add_assign(&mut self, rhs: TextUnit) {
        self.0 += rhs.0
    }
}

impl ops::Sub<TextUnit> for TextUnit {
    type Output = TextUnit;
    fn sub(self, rhs: TextUnit) -> TextUnit {
        TextUnit(self.0 - rhs.0)
    }
}

impl ops::SubAssign<TextUnit> for TextUnit {
    fn sub_assign(&mut self, rhs: TextUnit) {
        self.0 -= rhs.0
    }
}