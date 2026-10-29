// rustfmt-format_code_in_doc_comments: true

/// ```
/// # #[rustversion::since(1.36)]
/// # fn dox() {
/// # use std::pin::Pin;
/// # type Projection<'a> = &'a ();
/// # type ProjectionRef<'a> = &'a ();
/// # trait Dox {
/// fn   project_ex (self: Pin<&mut Self>) -> Projection<'_>;
/// fn   project_ref(self: Pin<&Self>) -> ProjectionRef<'_>;
/// # }
/// # }
/// ```
struct Foo;
