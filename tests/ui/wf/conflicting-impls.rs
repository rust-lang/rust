//@ edition: 2021

struct Ty;

impl TryFrom<Ty> for u8 {
    type Error = Ty;
    fn try_from(_: Ty) -> Result<Self, Self::Error> {
        //~^ ERROR type annotations needed
        loop {}
    }
}

impl TryFrom<Ty> for u8 {
    //~^ ERROR conflicting implementations of trait
    type Error = Ty;
    fn try_from(_: Ty) -> Result<Self, Self::Error> {
        //~^ ERROR type annotations needed
        loop {}
    }
}

fn main() {}
