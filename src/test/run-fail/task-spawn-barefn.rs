// error-pattern:Ensure that the child task runs by failing

fn main() {
    // the purpose of this test is to make sure that task::spawn()
    // works when provided with a bare function:
    task::spawn(startfn);
}

fn startfn() {
    assert str::is_empty("Ensure that the child task runs by failing");
}
