#![debugger_visualizer(natvis_file = "./foo.natvis")]
#![debugger_visualizer(gdb_script_file = "./foo.py")]

pub struct Foo {
    pub x: u32,
}
