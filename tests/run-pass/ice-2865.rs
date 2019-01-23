#[allow(dead_code)]
struct Ice {
    size: String,
}

impl<'a> From<String> for Ice {
    fn from(_: String) -> Self {
        let text = || "iceberg".to_string();
        Self { size: text() }
    }
}

fn main() {}
