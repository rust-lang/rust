struct Bug<S>{ //~ ERROR parameter `S` is never used [E0392]
    A: [(); {
        let x: [u8; Self::W] = [0; Self::W]; //~ ERROR generic `Self` types are currently not permitted in anonymous constants
        //~^ ERROR generic `Self` types are currently not permitted in anonymous constants
        //~^^ ERROR the size for values of type `S` cannot be known at compilation time [E0277]
        F //~ ERROR cannot find value `F` in this scope [E0425]
    }
} //~ ERROR mismatched closing delimiter: `}`
//~^ ERROR mismatched closing delimiter: `}`

fn main() {}
