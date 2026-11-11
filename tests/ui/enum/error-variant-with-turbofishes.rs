enum Struct<const N: usize> { Variant { x: [(); N] } }

fn test() {
    let x = Struct::<0>::Variant;
    //~^ ERROR expected value, found struct variant `Struct<0>::Variant`
}

fn main() {}
