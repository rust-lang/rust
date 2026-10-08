//@ has issue_85454/trait.TryFromBreak.html
//@ has - '//pre[@class="rust item-decl"]' 'pub trait TryFromBreak<R = <Self as Try>::Break> { fn from_break(residual: R) -> Self; }'
pub trait TryFromBreak<R = <Self as Try>::Break> {
    fn from_break(residual: R) -> Self;
}

pub trait Try: TryFromBreak {
    type Output;
    type Break;
    fn from_output(output: Self::Output) -> Self;
    fn branch(self) -> ControlFlow<Self::Break, Self::Output>;
}

pub enum ControlFlow<B, C = ()> {
    Continue(C),
    Break(B),
}
