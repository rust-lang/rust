//@ known-bug: #138510
fn main()
where
    #[repr()]
    _: Sized,
{
}
