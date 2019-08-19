// build-pass (FIXME(62277): could be check-pass?)

const PARSE_BOOL: Option<&'static str> = None;
static FOO: (Option<&str>, u32) = (PARSE_BOOL, 42);

fn main() {}
