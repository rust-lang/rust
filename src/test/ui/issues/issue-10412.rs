trait Serializable<'self, T> { //~ ERROR lifetimes cannot use keyword names
    fn serialize(val : &'self T) -> Vec<u8>; //~ ERROR lifetimes cannot use keyword names
    fn deserialize(repr : &[u8]) -> &'self T; //~ ERROR lifetimes cannot use keyword names
}

impl<'self> Serializable<str> for &'self str { //~ ERROR lifetimes cannot use keyword names
    //~^ ERROR lifetimes cannot use keyword names
    //~| ERROR implicit elided lifetime not allowed here
    //~| ERROR the size for values of type `str` cannot be known at compilation time
    fn serialize(val : &'self str) -> Vec<u8> { //~ ERROR lifetimes cannot use keyword names
        vec![1]
    }
    fn deserialize(repr: &[u8]) -> &'self str { //~ ERROR lifetimes cannot use keyword names
        "hi"
    }
}

fn main() {
    println!("hello");
    let x = "foo".to_string();
    let y = x;
    println!("{}", y);
}
