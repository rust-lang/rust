// Regression test for #71546.

// ignore-compare-mode-nll
// NLL stderr is different from the original one.

pub fn serialize_as_csv<V>(value: &V) -> Result<String, &str>
where
    V: 'static,
    for<'a> &'a V: IntoIterator,
    for<'a> <&'a V as IntoIterator>::Item: ToString + 'static,
{
    let csv_str: String = value //~ ERROR: the associated type `<&'a V as IntoIterator>::Item` may not live long enough
        .into_iter()
        .map(|elem| elem.to_string())
        .collect::<String>();
    Ok(csv_str)
}

fn main() {}
