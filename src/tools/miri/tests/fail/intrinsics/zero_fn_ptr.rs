//@normalize-stderr-test: "\|.*::abort\(\).*" -> "| ABORT()"
//@normalize-stderr-test: "\| +\^+" -> "| ^"
//@normalize-stderr-test: "\n +[0-9]+:[^\n]+" -> ""
//@normalize-stderr-test: "\n +at [^\n]+" -> ""
//@error-in-other-file: aborted execution

#[allow(deprecated, invalid_value)]
fn main() {
    let _ = unsafe { std::mem::zeroed::<fn()>() };
}
