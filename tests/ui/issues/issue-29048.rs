//@ check-pass
pub struct Chan;
pub struct ChanSelect<'c, T> {
    chans: Vec<(&'c Chan, T)>,
}
impl<'c, T> ChanSelect<'c, T> {
    pub fn add_recv_ret(&mut self, chan: &'c Chan, ret: T)
    {
        self.chans.push((chan, ret));
    }
}
fn main() {}
