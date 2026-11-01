fn generic<const N: u32>() {}

trait Collate<const A: u32> {
    type Pass;
    fn collate(self) -> Self::Pass;
}

impl<const B: u32> Collate<B> for i32 {
    type Pass = ();
    fn collate(self) -> Self::Pass {
        generic::<{ true }>()
        //~^ ERROR: mismatched types
    }
}

fn main() {}
