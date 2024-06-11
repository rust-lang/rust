//@ run-pass
//@ ignore-pass (JSON checks don't run under --check=pass)
//@ check-run-stdout-is-json-lines

//@ revisions: good bad_list bad_obj bad_empty bad_ws
//@ [bad_list] should-fail
//@ [bad_obj] should-fail
//@ [bad_empty] should-fail
//@ [bad_ws] should-fail

// Check that `//@ check-run-stdout-is-json-lines` allows valid JSON lines and
// rejects invalid JSON lines, even without `//@ check-run-results`.

fn main() {
    println!("true");
    println!(r#"[ "this is valid json" ]"#);
    println!(r#"{{ "key": "this is valid json" }}"#);

    if cfg!(bad_list) {
        println!(r#"[ "this is invalid json", ]"#);
    }
    if cfg!(bad_obj) {
        println!(r#"{{ "key": "this is invalid json", }}"#);
    }

    // Every line must be valid JSON, and a blank or whitespace-only string is
    // not valid JSON.
    if cfg!(bad_empty) {
        println!();
    }
    if cfg!(bad_ws) {
        println!(" \t \t ");
    }
}
