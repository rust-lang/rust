extern crate interesting;

fn main() {
    // function pointer case
    let fns: Vec<_> =
        std::iter::repeat(interesting::function_called_always as fn()).take(1000).collect();
    interesting::call_a_bunch_of_functions(&fns[..]);

    // Trait object case
    let trait_objects = vec![0u32; 1000];
    let trait_objects: Vec<_> = trait_objects.iter().map(|x| x as &dyn interesting::Foo).collect();
    interesting::call_a_bunch_of_trait_methods(&trait_objects[..]);
}
