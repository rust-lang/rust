//@ known-bug: #122903
impl Struct {
    fn box_box_ref_Struct(
        self: impl FnMut(Box<impl FnMut(&mut Self)>),
    ) -> &u32 {
        f
    }
}
