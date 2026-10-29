pub struct Struct {
    a: Vec<Struct>,
}

impl Struct {
    pub fn test(&self) {
        if let [Struct { a: [] }] = &self.a {
            //~^ ERROR expected an array or slice
            //~| ERROR expected an array or slice
            println!("matches!")
        }

        if let [Struct { a: [] }] = &self.a[..] {
            //~^ ERROR expected an array or slice
            println!("matches!")
        }
    }
}

fn main() {}
