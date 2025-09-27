//@ only-apple
//@ needs-dynamic-linking

#[link(name = "uwu", kind = "raw-dylib", modifiers = "+verbatim")]
//~^ ERROR: link kind `raw-dylib` is unstable on Mach-O platforms
unsafe extern "C" {
    safe fn kawaii();
}

fn main() {}
