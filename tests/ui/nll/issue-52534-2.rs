fn foo(x: &u32) -> &u32 {
    let y;

    {
        let x = 32;
        y = &x
//~^ ERROR does not live long enough
    }

    println!("{}", y);
    x
}

fn main() { }
