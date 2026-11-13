#![debugger_visualizer(gdb_script_file = "my_gdb_script.py")]

fn main() {
    const _UNUSED: u32 = {
        mod inner {
            #![debugger_visualizer(natvis_file = "my_visualizers/bar.natvis")]
            pub const XYZ: u32 = 123;
        }

        inner::XYZ + 1
    };
}
