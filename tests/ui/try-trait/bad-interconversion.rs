use std::ops::ControlFlow;

fn result_to_result() -> Result<u64, u8> {
    Ok(Err(123_i32)?)
    //~^ ERROR `?` couldn't convert the error to `u8`
}

fn option_to_result() -> Result<u64, String> {
    Some(3)?;
    //~^ ERROR the `?` operator can only be used on `Result`s, not `Option`s, in a function that returns `Result`
    Ok(10)
}

fn control_flow_to_result() -> Result<u64, String> {
    Ok(ControlFlow::Break(123)?)
    //~^ ERROR the `?` operator can only be used on `Result`s in a function that returns `Result`
}

fn result_to_option() -> Option<u16> {
    Some(Err("hello")?)
    //~^ ERROR the `?` operator can only be used on `Option`s, not `Result`s, in a function that returns `Option`
}

fn control_flow_to_option() -> Option<u64> {
    Some(ControlFlow::Break(123)?)
    //~^ ERROR the `?` operator can only be used on `Option`s in a function that returns `Option`
}

fn result_to_control_flow() -> ControlFlow<String> {
    ControlFlow::Continue(Err("hello")?)
    //~^ ERROR the `?` operator can only be used on `ControlFlow`s in a function that returns `ControlFlow`
}

fn option_to_control_flow() -> ControlFlow<u64> {
    Some(3)?;
    //~^ ERROR the `?` operator can only be used on `ControlFlow`s in a function that returns `ControlFlow`
    ControlFlow::Break(10)
}

fn control_flow_to_control_flow() -> ControlFlow<i64> {
    ControlFlow::Break(4_u8)?;
    //~^ ERROR the `?` operator in a function that returns `ControlFlow<B, _>` can only be used on other `ControlFlow<B, _>`s
    ControlFlow::Continue(())
}

fn main() {}
