// run-pass
#![allow(dead_code)]

#[derive(Debug)]
enum Token {
    Text(String),
    ETag(Vec<String>, String),
    UTag(Vec<String>, String),
    Section(Vec<String>, bool, Vec<Token>, String,
            String, String, String, String),
    IncompleteSection(Vec<String>, bool, String, bool),
    Partial(String, String, String),
}

fn check_strs(actual: &str, expected: &str) -> bool
{
    if actual != expected
    {
        println!("Found {}, but expected {}", actual, expected);
        return false;
    }
    return true;
}

pub fn main()
{
    let t = Token::Text("foo".to_string());
    let u = Token::Section(vec!["alpha".to_string()],
                    true,
                    vec![t],
                    "foo".to_string(),
                    "foo".to_string(), "foo".to_string(), "foo".to_string(),
                    "foo".to_string());
    let v = format!("{:?}", u);    // this is the line that causes the seg fault
    assert!(!v.is_empty());
}
