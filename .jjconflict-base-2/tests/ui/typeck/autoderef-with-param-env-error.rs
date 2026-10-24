fn foo()
where
    T: Send,
    //~^ ERROR cannot find type `T` in this scope
{
    let s = "abc".to_string();
}

fn main() {}
