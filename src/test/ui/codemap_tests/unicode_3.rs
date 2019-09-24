// build-pass (FIXME(62277): could be check-pass?)

fn main() {
    let s = "ZͨA͑ͦ͒͋ͤ͑̚L̄͑͋Ĝͨͥ̿͒̽̈́Oͥ͛ͭ!̏"; while true { break; }
    println!("{}", s);
}
