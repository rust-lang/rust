// @has tuple_struct_fields/struct.Tooople.html
// @has - //span '0: usize'
// @has - 'Wow! i love tuple fields!'
// @!has - 'I should be invisible'

pub struct Tooople(
    /// Wow! i love tuple fields!
    pub usize,
    /// I should be invisible
    u8,
);
