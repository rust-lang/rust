// Don't suggest importing a function from a private dependency.
// Issues: #138191, #142676

// Avoid suggesting traits from std-private deps
//@ forbid-output: compiler_builtins
//@ forbid-output: object

struct VecReader(Vec<u8>);

impl std::io::Read for VecReader {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.0.read(buf)
        //~^ ERROR no method named `read` found for struct `Vec<u8>`
    }
}

fn main() {
    let _ = u8::cast_from_lossy(9);
    //~^ ERROR no function or associated item named `cast_from_lossy` found for type `u8`
}
