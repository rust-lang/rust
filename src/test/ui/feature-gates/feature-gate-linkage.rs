extern "C" {
    #[linkage = "extern_weak"] static foo: isize;
    //~^ ERROR: the `linkage` attribute is experimental and not portable
}

fn main() {}
