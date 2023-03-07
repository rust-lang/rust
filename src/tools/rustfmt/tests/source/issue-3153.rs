// rustfmt-wrap_comments: true

/// This may panic if:
/// - there are fewer than `max_header_bytes` bytes preceding the body
/// - there are fewer than `max_footer_bytes` bytes following the body
/// - the sum of the body bytes and post-body bytes is less than the sum
///   of `min_body_and_padding_bytes` and `max_footer_bytes` (in other
///   words, the minimum body and padding byte requirement is not met)
fn foo() {}
