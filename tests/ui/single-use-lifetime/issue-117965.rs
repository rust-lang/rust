#![deny(single_use_lifetimes)]

pub enum Data<'a> {
    Borrowed(&'a str),
    Owned(String),
}

impl<'a> Data<'a> {
    pub fn get<'b: 'a>(&'b self) -> &'a str {
        //~^ ERROR lifetime parameter `'b` only used once
        match &self {
            Self::Borrowed(val) => val,
            Self::Owned(val) => &val,
        }
    }
}

fn main() {}
