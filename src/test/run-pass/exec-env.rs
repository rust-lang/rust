// xfail-fast (exec-env not supported in fast mode)
// exec-env:TEST_EXEC_ENV=22

fn main() {
    assert os::getenv(~"TEST_EXEC_ENV") == Some(~"22");
}
