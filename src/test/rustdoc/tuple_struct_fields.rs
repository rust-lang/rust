// @has tuple_struct_fields/struct.Tooople.html
// @has - //span '0: usize'
// @has - 'Wow! i love tuple fields!'

pub struct Tooople(
    /// Wow! i love tuple fields!
    pub usize,
    u8,
);
