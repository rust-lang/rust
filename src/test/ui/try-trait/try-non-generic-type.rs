// run-pass

#![feature(control_flow_enum)]
#![feature(try_trait_v2)]

use std::num::NonZeroI32;
use std::ops::{ControlFlow, Try, FromResidual};

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
#[repr(transparent)]
pub struct ResultCode(pub i32);
impl ResultCode {
    const SUCCESS: Self = ResultCode(0);
}

pub struct ResultCodeResidual(NonZeroI32);

#[derive(Debug, Clone)]
pub struct FancyError(String);

impl Try for ResultCode {
    type Ok = ();
    type Residual = ResultCodeResidual;
    fn branch(self) -> ControlFlow<Self::Residual> {
        match NonZeroI32::new(self.0) {
            Some(r) => ControlFlow::Break(ResultCodeResidual(r)),
            None => ControlFlow::Continue(()),
        }
    }
    fn from_output((): ()) -> Self {
        ResultCode::SUCCESS
    }
}

impl FromResidual for ResultCode {
    fn from_residual(r: ResultCodeResidual) -> Self {
        ResultCode(r.0.into())
    }
}

impl<T, E: From<FancyError>> FromResidual<ResultCodeResidual> for Result<T, E> {
    fn from_residual(r: ResultCodeResidual) -> Self {
        Err(FancyError(format!("Something fancy about {} at {:?}", r.0, std::time::SystemTime::now())).into())
    }
}

fn fine() -> ResultCode {
    ResultCode(0)
}

fn bad() -> ResultCode {
    ResultCode(-13)
}

fn i() -> ResultCode {
    fine()?;
    bad()?;
    ResultCode::SUCCESS
}

fn main() -> Result<(), FancyError> {
    assert_eq!(i(), ResultCode(-13));
    fine()?;
    Ok(())
}
