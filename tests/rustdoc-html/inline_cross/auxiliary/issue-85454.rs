//@ has issue_85454/trait.FromResidual.html
//@ has - '//pre[@class="rust item-decl"]' 'pub trait FromResidual<R = <Self as Try>::Residual> { fn from_residual(residual: R) -> Self; }'
pub trait FromResidual<R = <Self as Try>::Residual> {
    fn from_residual(residual: R) -> Self;
}

pub trait Try: FromResidual {
    type Output;
    type Residual;
    fn from_output(output: Self::Output) -> Self;
    fn branch(self) -> ControlFlow<Self::Residual, Self::Output>;
}

pub enum ControlFlow<B, C = ()> {
    Continue(C),
    Break(B),
}
