//@ normalize-stderr: "foo.random:.*\(" -> "foo.random: $$FILE_NOT_FOUND_MSG ("
//@ normalize-stderr: "os error \d+" -> "os error $$FILE_NOT_FOUND_CODE"

#![debugger_visualizer(random_file = "../foo.random")] //~ ERROR malformed `debugger_visualizer` attribute input
#![debugger_visualizer(natvis_file = "../foo.random")] //~ ERROR
fn main() {}
