// rustc --test ignores2.rs && ./ignores2
extern mod std;
use path::{Path};

type rsrc_loader = fn~ (path: &Path) -> result::Result<~str, ~str>;

#[test]
fn tester()
{
    let loader: rsrc_loader = |_path| {result::Ok(~"more blah")};

    let path = path::from_str("blah");
    assert loader(&path).is_ok();
}

fn main() {}
