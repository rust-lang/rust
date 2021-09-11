// run-pass

// Test binary_search_by_key lifetime. Issue #34683

#[allow(dead_code)]
#[derive(Debug)]
struct Assignment {
    topic: String,
    partition: i32,
}

fn main() {
    let xs = vec![
        Assignment { topic: "abc".into(), partition: 1 },
        Assignment { topic: "def".into(), partition: 2 },
        Assignment { topic: "ghi".into(), partition: 3 },
    ];

    let key: &str = "def";
    let r = xs.binary_search_by_key(&key, |e| &e.topic);
    assert_eq!(Ok(1), r.map(|i| i));
}
