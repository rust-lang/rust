// This is a companion to the similarly-named test in run-pass.
//
// It tests macros that unavoidably produce compile errors.

fn compile_error() {
    compile_error!("lel"); //~ ERROR lel
    compile_error!("lel",); //~ ERROR lel
}

fn main() {}
