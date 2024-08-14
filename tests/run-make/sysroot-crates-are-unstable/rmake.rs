use run_make_support::python_command;

fn main() {
    python_command().arg("test.py").run();
}
