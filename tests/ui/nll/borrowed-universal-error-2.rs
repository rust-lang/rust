fn foo<'a>(x: &'a (u32,)) -> &'a u32 {
    let v = 22;
    &v
    //~^ ERROR cannot return reference to local variable `v` [E0515]
}

fn main() {}
