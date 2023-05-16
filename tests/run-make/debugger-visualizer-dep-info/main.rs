#![debugger_visualizer(gdb_script_file = "foo.py")]

fn main() {
    const _UNUSED: u32 = {
        mod inner {
            #![debugger_visualizer(gdb_script_file = "my_visualizers/bar.py")]
            pub const XYZ: u32 = 123;
        }

        inner::XYZ + 1
    };
}
