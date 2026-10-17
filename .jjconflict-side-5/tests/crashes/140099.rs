//@ known-bug: #140099
struct a;
impl From for a where for<'any> &'any mut (): Clone {}
fn b() -> Result<(), std::convert::Infallible> {
    || -> Result<_, a> { b()? }
}
