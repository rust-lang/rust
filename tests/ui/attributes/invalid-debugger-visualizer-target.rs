#[debugger_visualizer(natvis_file = "./foo.natvis.xml")]
//~^ ERROR `#[debugger_visualizer]` attribute cannot be used on functions
fn main() {}
