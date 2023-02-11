// @has union/union.U.html
pub union U {
    // @has - //pre "pub a: u8"
    pub a: u8,
    // @has - //pre "/* private fields */"
    // @!has - //pre "b: u16"
    b: u16,
}
