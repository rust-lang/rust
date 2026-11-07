// Regression test for #71546.
//
// Made to pass as part of fixing #98095.
//
//@ check-pass

pub fn serialize_as_csv<V>(value: &V) -> Result<String, &str>
where
    V: 'static,
    for<'a> &'a V: IntoIterator,
    for<'a> <&'a V as IntoIterator>::Item: ToString + 'static,
{
    let csv_str: String = value.into_iter().map(|elem| elem.to_string()).collect::<String>();
    Ok(csv_str)
}

fn main() {}
