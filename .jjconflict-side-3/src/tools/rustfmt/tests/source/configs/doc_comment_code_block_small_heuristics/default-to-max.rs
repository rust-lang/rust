// rustfmt-format_code_in_doc_comments: true
// rustfmt-use_small_heuristics: Default
// rustfmt-doc_comment_code_block_small_heuristics: Max

/// Start of a doc comment.
///
/// ```
/// enum Lorem {
///     Ipsum,
///     Dolor(bool),
///     Sit { amet: Consectetur, adipiscing: Elit },
/// }
///
/// fn main() {
///     lorem(
///         "lorem",
///         "ipsum",
///         "dolor",
///         "sit",
///         "amet",
///         "consectetur",
///         "adipiscing",
///     );
///
///     let lorem = Lorem {
///         ipsum: dolor,
///         sit: amet,
///     };
///
///     let lorem = if ipsum { dolor } else { sit };
/// }
///
/// fn format_let_else() {
///     let Some(a) = opt else {};
///
///     let Some(b) = opt else { return };
///
///     let Some(c) = opt else { return };
///
///     let Some(d) = some_very_very_very_very_long_name else {
///         return;
///     };
/// }
/// ```
///
/// End of a doc comment.
struct S;

enum Lorem {
    Ipsum,
    Dolor(bool),
    Sit { amet: Consectetur, adipiscing: Elit },
}

fn main() {
    lorem("lorem", "ipsum", "dolor", "sit", "amet", "consectetur", "adipiscing");

    let lorem = Lorem { ipsum: dolor, sit: amet };

    let lorem = if ipsum { dolor } else { sit };
}

fn format_let_else() {
    let Some(a) = opt else {};

    let Some(b) = opt else { return };

    let Some(c) = opt else { return };

    let Some(d) = some_very_very_very_very_long_name else { return };
}
