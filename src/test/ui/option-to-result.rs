fn main(){ }

fn test_result() -> Result<(),()> {
    let a:Option<()> = Some(());
    a?;//~ ERROR the `?` operator can only be used in a function that returns `Result` or `Option`
    Ok(())
}

fn test_option() -> Option<i32>{
    let a:Result<i32, i32> = Ok(5);
    a?;//~ ERROR the `?` operator can only be used in a function that returns `Result` or `Option`
    Some(5)
}
