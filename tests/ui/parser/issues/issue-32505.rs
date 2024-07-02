pub fn test() {
    foo(|_|) //~ ERROR expected expression, found `)`
    //~^ ERROR cannot find function `foo`
}

fn main() { }
