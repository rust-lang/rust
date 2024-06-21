fn foo()
where
    T: Send,
    //~^ cannot find type `T`
{
    let s = "abc".to_string();
}

fn main() {}
