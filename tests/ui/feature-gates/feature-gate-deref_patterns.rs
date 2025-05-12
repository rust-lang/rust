fn main() {
    // We reuse the `box` syntax so this doesn't actually test the feature gate but eh.
    let box x = Box::new('c'); //~ ERROR box pattern syntax is experimental
    println!("x: {}", x);

    // `box` syntax is allowed to be cfg-ed out for historical reasons (#65742).
    #[cfg(false)]
    let box _x = Box::new('c');
}
