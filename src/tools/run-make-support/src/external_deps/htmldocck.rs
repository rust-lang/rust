use super::python::python_command;
use crate::command::Command;
use crate::source_root;

/// `htmldocck` is a python script which is used for rustdoc test suites, it is assumed to be
/// available at `$SOURCE_ROOT/src/etc/htmldocck.py`.
#[track_caller]
#[must_use]
pub fn htmldocck() -> Command {
    let mut python = python_command();
    python.arg(source_root().join("src/etc/htmldocck.py"));
    python
}
