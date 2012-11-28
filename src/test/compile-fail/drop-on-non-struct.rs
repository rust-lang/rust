type Foo = @[u8];

impl Foo : Drop {   //~ ERROR the Drop trait may only be implemented
    fn finalize(&self) {
        io::println("kaboom");
    }
}

fn main() {
}


