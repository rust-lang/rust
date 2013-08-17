struct StringBuffer {
    s: ~str
}

impl StringBuffer {
    pub fn append(&mut self, v: &str) {
        self.s.push_str(v);
    }
}

fn to_str(sb: StringBuffer) -> ~str {
    sb.s
}

fn main() {
    let mut sb = StringBuffer {s: ~""};
    sb.append("Hello, ");
    sb.append("World!");
    let str = to_str(sb);
    assert_eq!(str, ~"Hello, World!");
}
