fn main() {
    assert "[1, 2, 3]" == sys::log_str([1, 2, 3]);
    assert #fmt["%?/%5?", [1, 2, 3], "hi"] == "[1, 2, 3]/ \"hi\"";
}
