//@ compile-flags: -Zunstable-options -Zoffload=Device -Clto=fat

#![feature(gpu_offload)]

fn main() {
    // `args` is not a tuple literal.
    core::offload::offload! { kernel = kernel_0, args = 42 }
    //~^ ERROR `args` must be a tuple literal

    // FIXME(offload): Binding a tuple to a variable would currently bypass the
    // per-argument launch checks, so the macro rejects it for now.
    let args = ();
    core::offload::offload! { kernel = kernel_0, args = args }
    //~^ ERROR `args` must be a tuple literal
}

fn kernel_0() {}
