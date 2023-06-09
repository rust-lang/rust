fn negative()
where
    i32: !Copy,
{
}

fn maybe_const_negative()
where
    i32: ~const !Copy,
{
}
