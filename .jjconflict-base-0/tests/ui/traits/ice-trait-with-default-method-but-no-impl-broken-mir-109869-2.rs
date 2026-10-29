// ICE 'broken MIR: bad assignment: NoSolution'
// on trait with default method and no impls
// issue: rust-lang/rust#109869

trait Empty<T> {}

impl<T> Default for dyn Empty<T>
where
    Self: Sized,
{
    fn default() -> Self {
        ()
        //~^ ERROR mismatched types
    }
}

pub fn main() {}
