#![allow(unused_assignments)]
// expect-exit-status-1

fn call(return_error: bool) -> Result<(),()> {
    if return_error {
        Err(())
    } else {
        Ok(())
    }
}

fn main() -> Result<(),()> {
    let mut
        countdown = 10
    ;
    for
        _
    in
        0..10
    {
        countdown
            -= 1
        ;
        if
            countdown < 5
        {
            call(/*return_error=*/ true)?;
            call(/*return_error=*/ false)?;
        }
        else
        {
            call(/*return_error=*/ false)?;
        }
    }
    Ok(())
}
