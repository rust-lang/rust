//@ known-bug: #140100
fn a()
where
    b: Sized,
{
    println!()
}
