// Regression test for issue #97933.
//
// Test that certain unevaluated constant expressions that are
// deemed too verbose or complex and that may leak private or
// `doc(hidden)` struct fields are not displayed in the documentation.
//
// Read the documentation of `rustdoc::clean::utils::print_const_expr`
// for further details.

//@ has hide_complex_unevaluated_consts/trait.Container.html
pub trait Container {
    // A helper constant that prevents const expressions containing it
    // from getting fully evaluated since it doesn't have a body and
    // thus is non-reducible. This allows us to specifically test the
    // pretty-printing of *unevaluated* consts.
    const ABSTRACT: i32;

    // Ensure that the private field does not get leaked:
    //
    //@ has - '//*[@id="associatedconstant.STRUCT0"]' \
    //        'const STRUCT0: Struct = _'
    const STRUCT0: Struct = Struct { private: () };

    //@ has - '//*[@id="associatedconstant.STRUCT1"]' \
    //        'const STRUCT1: (Struct,) = _'
    const STRUCT1: (Struct,) = (Struct{private: /**/()},);

    // Although the struct field is public here, check that it is not
    // displayed. In a future version of rustdoc, we definitely want to
    // show it. However for the time being, the printing logic is a bit
    // conservative.
    //
    //@ has - '//*[@id="associatedconstant.STRUCT2"]' \
    //        'const STRUCT2: Record = _'
    const STRUCT2: Record = Record { public: 5 };

    // Test that we do not show the incredibly verbose match expr:
    //
    //@ has - '//*[@id="associatedconstant.MATCH0"]' \
    //        'const MATCH0: i32 = _'
    const MATCH0: i32 = match 234 {
        0 => 1,
        _ => Self::ABSTRACT,
    };

    //@ has - '//*[@id="associatedconstant.MATCH1"]' \
    //        'const MATCH1: bool = _'
    const MATCH1: bool = match Self::ABSTRACT {
        _ => true,
    };

    // Check that we hide complex (arithmetic) operations.
    // In this case, it is a bit unfortunate since the expression
    // is not *that* verbose and it might be quite useful to the reader.
    //
    // However in general, the expression might be quite large and
    // contain match expressions and structs with private fields.
    // We would need to recurse over the whole expression and even more
    // importantly respect operator precedence when pretty-printing
    // the potentially partially censored expression.
    // For now, the implementation is quite simple and the choices
    // rather conservative.
    //
    //@ has - '//*[@id="associatedconstant.ARITH_OPS"]' \
    //        'const ARITH_OPS: i32 = _'
    const ARITH_OPS: i32 = Self::ABSTRACT * 2 + 1;
}

pub struct Struct { private: () }

pub struct Record { pub public: i32 }
