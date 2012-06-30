// pp-exact

// some single-line non-doc comment

/// some single line outer-docs
fn a() { }

fn b() {
    //! some single line inner-docs
}

/*
 * some multi-line non-doc comment
 */

/**
 * some multi-line outer-docs
 */
fn c() { }

fn d() {
    /*!
     * some multi-line inner-docs
     */
}

#[doc = "unsugared outer doc-comments work also"]
fn e() { }

fn f() {
    #[doc = "as do inner ones"];
}
