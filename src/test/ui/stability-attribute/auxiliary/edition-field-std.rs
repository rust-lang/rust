#![feature(staged_api)]
#![stable(feature = "intersperse", since = "1.0.0", edition = "2021")]

#[stable(feature = "intersperse", since = "1.0.0", edition = "2021")]
pub struct Intersperse;

#[stable(feature = "intersperse", since = "1.0.0", edition = "2021")]
pub trait Iterator {
    #[stable(feature = "intersperse", since = "1.0.0", edition = "2021")]
    fn intersperse(&self, separator: ()) -> Intersperse {
        unimplemented!()
    }
}
