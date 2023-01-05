fn g()
where
    'static: 'static,
    dyn 'static: 'static + Copy, //~ ERROR at least one trait is required for an object type
{
}

fn main() {}
