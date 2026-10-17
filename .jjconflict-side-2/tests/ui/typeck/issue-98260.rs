fn main() {}
trait A {
    fn a(aa: B) -> Result<_, B> {
    //~^ ERROR: the placeholder `_` is not allowed within types on item signatures for return types [E0121]
        Ok(())
    }
}

enum B {}
