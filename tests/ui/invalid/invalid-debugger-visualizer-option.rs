// normalize-stderr-test: "foo.random:.*\(" -> "foo.random: $$FILE_NOT_FOUND_MSG ("
// normalize-stderr-test: "os error \d+" -> "os error $$FILE_NOT_FOUND_CODE"

#![debugger_visualizer(random_file = "../foo.random")] //~ ERROR invalid argument
#![debugger_visualizer(natvis_file = "../foo.random")] //~ ERROR
fn main() {}
