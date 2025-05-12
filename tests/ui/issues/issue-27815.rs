mod A {}

fn main() {
    let u = A { x: 1 }; //~ ERROR expected struct, variant or union type, found module `A`
    let v = u32 { x: 1 }; //~ ERROR expected struct, variant or union type, found builtin type `u32`
    match () {
        A { x: 1 } => {}
        //~^ ERROR expected struct, variant or union type, found module `A`
        u32 { x: 1 } => {}
        //~^ ERROR expected struct, variant or union type, found builtin type `u32`
    }
}
