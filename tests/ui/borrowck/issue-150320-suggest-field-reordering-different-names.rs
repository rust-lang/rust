// Test for issue #150320 with different field names
// to verify the suggestion uses actual variable names, not hardcoded ones.

struct Config {
    path: String,
    length: usize,
}

impl Config {
    fn new(path: String) -> Self {
        Self {
            path,
            length: path.len(), //~ ERROR borrow of moved value: `path`
        }
    }
}

struct Data {
    name: String,
    size: usize,
}

impl Data {
    fn create(name: String) -> Self {
        Self {
            name,
            size: name.len(), //~ ERROR borrow of moved value: `name`
        }
    }
}

fn main() {
    let _ = Config::new("/tmp/file".to_string());
    let _ = Data::create("test".to_string());
}
