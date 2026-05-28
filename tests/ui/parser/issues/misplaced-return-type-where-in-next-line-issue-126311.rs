fn foo<T, K>()
//~^ HELP place the return type after the function parameters
where
    T: Default,
    K: Clone, -> Result<u8, String>
//~^ ERROR return type should be specified after the function parameters
{
    Ok(0)
}

fn main() {}
