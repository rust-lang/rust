use super::*;

#[test]
fn basic_find() {
    use SelectorPart::*;

    let j = serde_json::json!({
        "index": {
            "4": {
                "inner": {
                    "items": ["1", "2", "3"]
                }
            }
        }
    });

    let sel = find_selector(&j, &serde_json::json!("1"));
    let exp: Vec<Vec<SelectorPart>> = vec![vec![
        Field("index".to_owned()),
        Field("4".to_owned()),
        Field("inner".to_owned()),
        Field("items".to_owned()),
        Index(0),
    ]];

    assert_eq!(exp, sel);
}
