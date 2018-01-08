use super::*;

pub(super) fn literal(p: &mut Parser) -> bool {
    let literals = [
        TRUE_KW, FALSE_KW,
        INT_NUMBER, FLOAT_NUMBER,
        BYTE, CHAR,
        STRING, RAW_STRING, BYTE_STRING, RAW_BYTE_STRING,
    ];
    node_if(p, AnyOf(&literals), LITERAL, |_| ())
}