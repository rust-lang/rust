fn main(){ }

fn test_result() -> Result<(),()> {
    let a:Option<()> = Some(());
    a?;//~ ERROR `?` couldn't convert the error
    Ok(())
}

fn test_option() -> Option<i32>{
    let a:Result<i32, i32> = Ok(5);
    a?;//~ ERROR `?` couldn't convert the error
    Some(5)
}
