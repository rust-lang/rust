//@ edition: 2021

struct Ty;

impl TryFrom<Ty> for u8 {
    type Error = Ty;
    fn try_from(_: Ty) -> Result<Self, Self::Error> {
        loop {}
    }
}

impl TryFrom<Ty> for u8 {
    //~^ ERROR conflicting implementations of trait
    type Error = Ty;
    fn try_from(_: Ty) -> Result<Self, Self::Error> {
        loop {}
    }
}

fn main() {}
