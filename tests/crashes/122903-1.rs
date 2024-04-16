//@ known-bug: #122903
impl Struct {
    async fn box_box_ref_Struct(
        self: Box<Box<Self, impl FnMut(&mut Box<Box<Self, impl FnMut(&mut Self)>>)>>,
    ) -> &u32 {
        f
    }
}
