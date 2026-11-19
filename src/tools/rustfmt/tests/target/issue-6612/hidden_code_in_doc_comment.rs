//@ rustfmt-format_code_in_doc_comments: true
//@ rustfmt-style_edition: 2024

/// ```
/// println!("1"); // comment
/// # println!("2");
/// ```
struct S;

/// println!("1"); // comment
/// # println!("2");
struct S2;

/// ```
/// println!("1"); // comment
/// # println!("2");
/// ```
struct S3;

/// ```
/// println!("1"); // comment
///     # println!("2");
/// ```
struct S4;
