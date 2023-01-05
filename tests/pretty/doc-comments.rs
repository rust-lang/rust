// compile-flags: --crate-type=lib

// pp-exact

// some single-line non-doc comment

/// some single line outer-docs
fn a() {}

fn b() {
    //! some single line inner-docs
}

//////////////////////////////////
// some single-line non-doc comment preceded by a separator

//////////////////////////////////
/// some single-line outer-docs preceded by a separator
/// (and trailing whitespaces)
fn c() {}

/*
 * some multi-line non-doc comment
 */

/**
 * some multi-line outer-docs
 */
fn d() {}

fn e() {
    /*!
     * some multi-line inner-docs
     */
}

/********************************/
/*
 * some multi-line non-doc comment preceded by a separator
 */

/********************************/
/**
 * some multi-line outer-docs preceded by a separator
 */
fn f() {}

#[doc = "unsugared outer doc-comments work also"]
fn g() {}

fn h() {
    #![doc = "as do inner ones"]
}
