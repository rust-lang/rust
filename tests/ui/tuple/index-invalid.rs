//@ reference: expr.tuple-index.index-name-operand
//@ reference: expr.tuple-index.index-syntax
//@ reference: lex.token.literal.int.tuple-field.eq
//@ reference: type.tuple.field-name
fn main() {
    let _ = (((),),).1.0; //~ ERROR no field `1` on type `(((),),)`

    let _ = (((),),).0.1; //~ ERROR no field `1` on type `((),)`

    let _ = (((),),).000.000; //~ ERROR no field `000` on type `(((),),)`
}
