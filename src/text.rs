use std::fmt;
use std::ops;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TextUnit(u32);

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

impl From<u32> for TextUnit {
    fn from(tu: u32) -> TextUnit {
        TextUnit::new(tu)
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


#[derive(Clone, Copy, PartialEq, Eq)]
pub struct TextRange {
    start: TextUnit,
    end: TextUnit,
}

impl fmt::Debug for TextRange {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        <Self as fmt::Display>::fmt(self, f)
    }
}

impl fmt::Display for TextRange {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[{}; {})", self.start(), self.end())
    }
}


impl TextRange {
    pub fn empty() -> TextRange {
        TextRange::from_to(TextUnit::new(0), TextUnit::new(0))
    }

    pub fn from_to(from: TextUnit, to: TextUnit) -> TextRange {
        assert!(from <= to, "Invalid text range [{}; {})", from, to);
        TextRange { start: from, end: to }
    }

    pub fn from_len(from: TextUnit, len: TextUnit) -> TextRange {
        TextRange::from_to(from, from + len)
    }

    pub fn start(&self) -> TextUnit {
        self.start
    }

    pub fn end(&self) -> TextUnit {
        self.end
    }

    pub fn len(&self) -> TextUnit {
        self.end - self.start
    }

    pub fn is_empty(&self) -> bool {
        self.start() == self.end()
    }
}

impl ops::Index<TextRange> for str {
    type Output = str;

    fn index(&self, index: TextRange) -> &str {
        &self[index.start().0 as usize..index.end().0 as usize]
    }
}