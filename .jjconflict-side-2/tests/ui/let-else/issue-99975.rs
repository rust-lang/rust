//@ run-pass
//@ compile-flags: -C opt-level=3 -Zvalidate-mir



fn return_result() -> Option<String> {
    Some("ok".to_string())
}

fn start() -> String {
    let Some(content) = return_result() else {
        return "none".to_string()
    };

    content
}

fn main() {
    start();
}
