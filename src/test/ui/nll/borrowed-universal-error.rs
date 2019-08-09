fn gimme(x: &(u32,)) -> &u32 {
    &x.0
}

fn foo<'a>(x: &'a (u32,)) -> &'a u32 {
    let v = 22;
    gimme(&(v,))
    //~^ ERROR cannot return value referencing temporary value [E0515]
}

fn main() {}
