use core::ops::ControlFlow;

/// Similar to the `Try` trait, but also implemented for `()`.
pub trait VisitorResult {
    type Residual;
    fn output() -> Self;
    fn from_residual(residual: Self::Residual) -> Self;
    fn from_branch(b: ControlFlow<Self::Residual>) -> Self;
    fn branch(self) -> ControlFlow<Self::Residual>;
}

impl VisitorResult for () {
    #[cfg(feature = "nightly")]
    type Residual = !;

    #[cfg(not(feature = "nightly"))]
    type Residual = core::convert::Infallible;

    fn output() -> Self {}
    fn from_residual(_: Self::Residual) -> Self {}
    fn from_branch(_: ControlFlow<Self::Residual>) -> Self {}
    fn branch(self) -> ControlFlow<Self::Residual> {
        ControlFlow::Continue(())
    }
}

impl<T> VisitorResult for ControlFlow<T> {
    type Residual = T;

    fn output() -> Self {
        ControlFlow::Continue(())
    }
    fn from_residual(residual: Self::Residual) -> Self {
        ControlFlow::Break(residual)
    }
    fn from_branch(b: Self) -> Self {
        b
    }
    fn branch(self) -> Self {
        self
    }
}

#[macro_export]
macro_rules! try_visit {
    ($e:expr) => {
        match $crate::visit::VisitorResult::branch($e) {
            core::ops::ControlFlow::Continue(()) => (),
            #[allow(unreachable_code)]
            core::ops::ControlFlow::Break(r) => {
                return $crate::visit::VisitorResult::from_residual(r);
            }
        }
    };
}

#[macro_export]
macro_rules! visit_opt {
    ($visitor: expr, $method: ident, $opt: expr $(, $($extra_args: expr),* )?) => {
        if let Some(x) = $opt {
            $crate::try_visit!($visitor.$method(x $(, $($extra_args,)* )?));
        }
    }
}

#[macro_export]
macro_rules! walk_list {
    ($visitor: expr, $method: ident, $list: expr $(, $($extra_args: expr),* )?) => {
        for elem in $list {
            $crate::try_visit!($visitor.$method(elem $(, $($extra_args,)* )?));
        }
    }
}

#[macro_export]
macro_rules! walk_visitable_list {
    ($visitor: expr, $list: expr $(, $($extra_args: expr),* )?) => {
        for elem in $list {
            $crate::try_visit!(elem.visit_with($visitor $(, $($extra_args,)* )?));
        }
    }
}
