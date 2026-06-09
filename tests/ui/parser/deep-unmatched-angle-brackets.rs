trait Mul<T> {
    type Output;
}
trait Matrix: Mul<<Self as Matrix>::Row, Output = ()> {
    type Row;
    type Transpose: Matrix<Row = Self::Row>;
}
fn is_mul<S, T: Mul<S, Output = ()>>() {}
fn f<T: Matrix>() {
    is_mul::<f::<f::<f::<f::<f::<f::<f::<f::<f::<f::<f::<f::<f::<f::<f::<f::<f::<f::<f::<f::<f::<
        f::<f::<f::<f::<f::<f::<f::<f::<f::<f::<f::<f::<f::<f::<f::<f::<f::<f::<f::<f::<f::<f::<
        f::<f::<f::<f::<f::<f::<f::<f::<f::<f::<f::<f::<f::<f::<f::<f::<f::<f::<f::<f::<f::<f::<
        f::<f::<f::<f::<f::<f::<f::<f::<f::<f::<f::<f::<f::<f::<f::<f::<f::<f::<f::<f::<f::<f::
        <f::<f::<f::<f::<f::<f::<f::<f::<f::<f::<f::<f::<f::<>();
    //~^ ERROR expected one of `!`, `+`, `,`, `::`, or `>`, found `(`
}
fn main() {}
