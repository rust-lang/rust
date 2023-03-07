fn main() {
    #[cfg(debug_assertions)]
    {
        println!("DEBUG");
    }
}

fn main() {
    #[cfg(feature = "foo")]
    {
        /*
        let foo = 0
        */
    }
}

fn main() {
    #[cfg(feature = "foo")]
    { /* let foo = 0; */ }
}

fn main() {
    #[foo]
    #[bar]
    #[baz]
    {
        // let foo = 0;
    }
}
