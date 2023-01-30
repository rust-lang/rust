// Checks whether shadowing a const parameter leads to an ICE (#85348).

impl<const N: usize> ArrayWindowsExample {
//~^ ERROR: cannot find type `ArrayWindowsExample` in this scope [E0412]
    fn next() {
        let mut N;
        //~^ ERROR: let bindings cannot shadow const parameters [E0530]
        //~| ERROR: type annotations needed [E0282]
    }
}

fn main() {}
