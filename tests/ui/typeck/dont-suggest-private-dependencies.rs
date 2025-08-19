// Don't suggest importing a function from a private dependency.
// Issues: #138191, #142676

// Avoid suggesting traits from std-private deps
//@ forbid-output: compiler_builtins
//@ forbid-output: object

// Check a custom trait to withstand changes in above crates
//@ aux-crate:public_dep=public-dep.rs
//@ compile-flags: -Zunstable-options
//@ forbid-output: private_dep

// By default, the `read` diagnostic suggests `std::os::unix::fs::FileExt::read_at`. Add
// something more likely to be recommended to make the diagnostic cross-platform.
trait DecoyRead {
    fn read1(&self) {}
}
impl<T> DecoyRead for Vec<T> {}

struct VecReader(Vec<u8>);

impl std::io::Read for VecReader {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.0.read(buf)
        //~^ ERROR no method named `read` found for struct `Vec<u8>`
    }
}

extern crate public_dep;
use public_dep::B;

fn main() {
    let _ = u8::cast_from_lossy(9);
    //~^ ERROR no function or associated item named `cast_from_lossy` found for type `u8`
    let _ = B::foo();
    //~^ ERROR no function or associated item named `foo` found for struct `B`
}
