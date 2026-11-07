//@ known-bug: #150517
trait Stream {
    type Item;
    fn next(self) -> ();
}
impl Stream for &'a () {}
impl<'a, A> Stream for <&A as Stream>::Item {}
trait StreamExt {
    fn f(self) -> ()
    where
        for<'b> &'b A: Stream,
    {
        self.next()
    }
}
