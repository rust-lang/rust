// Regression test for #71546.

pub fn serialize_as_csv<V>(value: &V) -> Result<String, &str>
where
    V: 'static,
    for<'a> &'a V: IntoIterator,
    for<'a> <&'a V as IntoIterator>::Item: ToString + 'static,
{
    let csv_str: String = value
        //~^ ERROR higher-ranked lifetime error
        //~| ERROR higher-ranked lifetime error
        //~| ERROR higher-ranked lifetime error
        .into_iter()
        .map(|elem| elem.to_string())
        //~^ ERROR higher-ranked lifetime error
        .collect::<String>();
        //~^ ERROR higher-ranked lifetime error
    Ok(csv_str)
}

fn main() {}
