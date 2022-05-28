#![feature(debugger_visualizer)]
#![debugger_visualizer(random_file = "../foo.random")] //~ ERROR invalid argument

fn main() {}
