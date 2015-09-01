// rustfmt-where_pred_indent: Inherit
// Test different indents.

fn qux()
    where X: TTTTTTTTTTTTTTTTTTTTTTTTTTTT,
    X: TTTTTTTTTTTTTTTTTTTTTTTTTTTT,
    X: TTTTTTTTTTTTTTTTTTTTTTTTTTTT,
    X: TTTTTTTTTTTTTTTTTTTTTTTTTTTT
{
    baz();
}
